"""Baseline Predictor — generates player predictions from historical averages.

No ML model training required. Uses the player's recent stats
(last N games or season averages) to project next-game performance.

This provides immediate predictions for every player in the database
without requiring any trained model or user input.
"""

import logging
from datetime import datetime
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from backend.db.models import Player, PlayerStat, Prediction, ModelVersion

logger = logging.getLogger(__name__)

# Weights for recency: more recent games weigh more
RECENCY_WINDOW = 30  # Look at last 30 games
MIN_GAMES_FOR_PREDICTION = 3


def _ensure_baseline_model_version(session: Session) -> ModelVersion:
    """Create or retrieve the 'baseline' model version entry."""
    mv = (
        session.query(ModelVersion)
        .filter_by(model_type="baseline", version="v1.0-historical-avg")
        .first()
    )
    if mv is None:
        mv = ModelVersion(
            model_type="baseline",
            version="v1.0-historical-avg",
            hyperparams_json={
                "method": "weighted_moving_average",
                "window": RECENCY_WINDOW,
                "description": "Predictions based on weighted historical averages. No ML training required.",
            },
            train_mse=0.0,
            val_mse=0.0,
            train_r2=0.0,
            val_r2=0.0,
            trained_at=datetime.utcnow(),
            checkpoint_path=None,
        )
        session.add(mv)
        session.commit()
        logger.info("Created baseline model version entry.")
    return mv


def generate_player_prediction(session: Session, player: Player, model_version: ModelVersion) -> dict | None:
    """Generate a prediction for a single player based on their historical stats.

    Uses a weighted average of recent games. More recent games get higher weight.
    Returns None if insufficient data.
    """
    # Fetch recent stats ordered by date (most recent first)
    stats = (
        session.query(PlayerStat)
        .filter(PlayerStat.player_id == player.id)
        .order_by(desc(PlayerStat.game_date))
        .limit(RECENCY_WINDOW)
        .all()
    )

    if len(stats) < MIN_GAMES_FOR_PREDICTION:
        return None

    # Calculate weighted averages
    # Weight: most recent game = N, second most recent = N-1, etc.
    total_weight = 0
    weighted_hits = 0.0
    weighted_hr = 0.0
    weighted_rbi = 0.0
    weighted_walks = 0.0

    for i, stat in enumerate(stats):
        weight = len(stats) - i  # Most recent gets highest weight
        total_weight += weight

        weighted_hits += (stat.hits or 0) * weight
        weighted_hr += (stat.home_runs or 0) * weight
        weighted_rbi += (stat.rbi or 0) * weight
        weighted_walks += (stat.walks or 0) * weight

    if total_weight == 0:
        return None

    pred_hits = round(weighted_hits / total_weight, 3)
    pred_hr = round(weighted_hr / total_weight, 3)
    pred_rbi = round(weighted_rbi / total_weight, 3)
    pred_walks = round(weighted_walks / total_weight, 3)

    # Confidence based on sample size (more games = higher confidence)
    confidence = min(len(stats) / RECENCY_WINDOW, 1.0)

    return {
        "predicted_hits": pred_hits,
        "predicted_hr": pred_hr,
        "predicted_rbi": pred_rbi,
        "predicted_walks": pred_walks,
        "confidence": round(confidence, 2),
    }


def generate_all_predictions(session: Session) -> int:
    """Generate baseline predictions for ALL players in the database.

    Returns the number of predictions generated.
    """
    mv = _ensure_baseline_model_version(session)

    players = session.query(Player).all()
    count = 0

    for player in players:
        pred_data = generate_player_prediction(session, player, mv)
        if pred_data is None:
            continue

        # Check if a recent baseline prediction already exists (within last hour)
        from datetime import timedelta
        recent_cutoff = datetime.utcnow() - timedelta(hours=1)
        existing = (
            session.query(Prediction)
            .filter(
                Prediction.player_id == player.id,
                Prediction.model_version_id == mv.id,
                Prediction.created_at >= recent_cutoff,
            )
            .first()
        )
        if existing:
            # Update instead of creating duplicate
            existing.predicted_hits = pred_data["predicted_hits"]
            existing.predicted_hr = pred_data["predicted_hr"]
            existing.predicted_rbi = pred_data["predicted_rbi"]
            existing.predicted_walks = pred_data["predicted_walks"]
            existing.confidence = pred_data["confidence"]
            existing.created_at = datetime.utcnow()
        else:
            pred = Prediction(
                player_id=player.id,
                model_version_id=mv.id,
                predicted_hits=pred_data["predicted_hits"],
                predicted_hr=pred_data["predicted_hr"],
                predicted_rbi=pred_data["predicted_rbi"],
                predicted_walks=pred_data["predicted_walks"],
                confidence=pred_data["confidence"],
                created_at=datetime.utcnow(),
            )
            session.add(pred)

        count += 1
        if count % 50 == 0:
            session.commit()
            logger.info(f"  Generated {count} baseline predictions...")

    session.commit()
    logger.info(f"Finished: generated {count} baseline predictions for {len(players)} players.")
    return count


def get_or_generate_prediction(session: Session, player: Player) -> dict | None:
    """Get existing prediction or generate one on-the-fly.

    Used by the game predictions endpoint to ensure every player
    has a prediction without requiring explicit model training.
    """
    mv = _ensure_baseline_model_version(session)

    # Check for any existing prediction
    existing = (
        session.query(Prediction)
        .filter(Prediction.player_id == player.id)
        .order_by(desc(Prediction.created_at))
        .first()
    )
    if existing:
        return {
            "predicted_hits": existing.predicted_hits,
            "predicted_hr": existing.predicted_hr,
            "predicted_rbi": existing.predicted_rbi,
            "predicted_walks": existing.predicted_walks,
            "confidence": existing.confidence,
        }

    # No prediction exists — generate one on-the-fly
    pred_data = generate_player_prediction(session, player, mv)
    if pred_data is None:
        return None

    # Store it for next time
    pred = Prediction(
        player_id=player.id,
        model_version_id=mv.id,
        predicted_hits=pred_data["predicted_hits"],
        predicted_hr=pred_data["predicted_hr"],
        predicted_rbi=pred_data["predicted_rbi"],
        predicted_walks=pred_data["predicted_walks"],
        confidence=pred_data["confidence"],
        created_at=datetime.utcnow(),
    )
    session.add(pred)
    session.commit()

    return pred_data
