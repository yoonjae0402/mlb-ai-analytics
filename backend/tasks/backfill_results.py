"""Backfill prediction results — compare predictions vs actual outcomes.

After games complete, fetch actual player stats from the DB and
compute error metrics for each prediction.
"""

import logging
from datetime import datetime, timedelta, date
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func

from backend.db.models import (
    Prediction, PredictionResult, PlayerStat, Player,
)

logger = logging.getLogger(__name__)


def backfill_prediction_results(session: Session, lookback_days: int = 7) -> int:
    """Compare predictions against actual stats for completed games.

    For each prediction that doesn't have a result yet, look for
    actual game stats within a reasonable date window.

    Returns the number of results created.
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    # Find predictions without results that are old enough to have actuals
    predictions = (
        session.query(Prediction)
        .outerjoin(PredictionResult)
        .filter(
            PredictionResult.id.is_(None),
            Prediction.created_at >= cutoff,
            Prediction.created_at < datetime.utcnow() - timedelta(hours=12),
        )
        .all()
    )

    if not predictions:
        logger.info("No predictions to backfill.")
        return 0

    logger.info(f"Found {len(predictions)} predictions to evaluate.")
    count = 0

    for pred in predictions:
        # Find the actual game stats for this player on the prediction date
        pred_date = pred.created_at.date()

        # Look for stats on the prediction date or the next day
        actual = (
            session.query(PlayerStat)
            .filter(
                PlayerStat.player_id == pred.player_id,
                PlayerStat.game_date.between(pred_date, pred_date + timedelta(days=1)),
            )
            .order_by(PlayerStat.game_date)
            .first()
        )

        if actual is None:
            continue

        # Compute per-prediction MSE
        actual_hits = actual.hits or 0
        actual_hr = actual.home_runs or 0
        actual_rbi = actual.rbi or 0
        actual_walks = actual.walks or 0

        pred_hits = pred.predicted_hits or 0
        pred_hr = pred.predicted_hr or 0
        pred_rbi = pred.predicted_rbi or 0
        pred_walks = pred.predicted_walks or 0

        mse = (
            (pred_hits - actual_hits) ** 2
            + (pred_hr - actual_hr) ** 2
            + (pred_rbi - actual_rbi) ** 2
            + (pred_walks - actual_walks) ** 2
        ) / 4.0

        result = PredictionResult(
            prediction_id=pred.id,
            actual_hits=actual_hits,
            actual_hr=actual_hr,
            actual_rbi=actual_rbi,
            actual_walks=actual_walks,
            mse=round(mse, 6),
            evaluated_at=datetime.utcnow(),
        )
        session.add(result)
        count += 1

        if count % 50 == 0:
            session.commit()

    session.commit()
    logger.info(f"Created {count} prediction results.")
    return count


def get_accuracy_summary(session: Session) -> dict:
    """Compute overall prediction accuracy metrics.

    Returns summary stats: avg MSE, MAE, hit rate, etc.
    """
    results = (
        session.query(PredictionResult)
        .join(Prediction)
        .all()
    )

    if not results:
        return {
            "total_evaluated": 0,
            "avg_mse": None,
            "avg_mae": None,
            "hit_rate": None,
            "per_stat": {},
        }

    total = len(results)

    # Per-stat errors
    hits_errors = []
    hr_errors = []
    rbi_errors = []
    walks_errors = []
    hits_correct = 0

    for r in results:
        pred = r.prediction
        p_hits = pred.predicted_hits or 0
        p_hr = pred.predicted_hr or 0
        p_rbi = pred.predicted_rbi or 0
        p_walks = pred.predicted_walks or 0

        a_hits = r.actual_hits or 0
        a_hr = r.actual_hr or 0
        a_rbi = r.actual_rbi or 0
        a_walks = r.actual_walks or 0

        hits_errors.append(abs(p_hits - a_hits))
        hr_errors.append(abs(p_hr - a_hr))
        rbi_errors.append(abs(p_rbi - a_rbi))
        walks_errors.append(abs(p_walks - a_walks))

        # "Hit rate" — prediction within 1 of actual for hits
        if abs(round(p_hits) - a_hits) <= 1:
            hits_correct += 1

    avg_mse = sum(r.mse for r in results) / total
    all_mae = [
        (abs((r.prediction.predicted_hits or 0) - (r.actual_hits or 0))
         + abs((r.prediction.predicted_hr or 0) - (r.actual_hr or 0))
         + abs((r.prediction.predicted_rbi or 0) - (r.actual_rbi or 0))
         + abs((r.prediction.predicted_walks or 0) - (r.actual_walks or 0))) / 4.0
        for r in results
    ]
    avg_mae = sum(all_mae) / total

    return {
        "total_evaluated": total,
        "avg_mse": round(avg_mse, 4),
        "avg_mae": round(avg_mae, 4),
        "hit_rate": round(hits_correct / total, 4) if total > 0 else None,
        "per_stat": {
            "hits": {
                "mae": round(sum(hits_errors) / total, 4),
                "within_1": round(hits_correct / total, 4),
            },
            "home_runs": {
                "mae": round(sum(hr_errors) / total, 4),
            },
            "rbi": {
                "mae": round(sum(rbi_errors) / total, 4),
            },
            "walks": {
                "mae": round(sum(walks_errors) / total, 4),
            },
        },
    }


def get_accuracy_by_player(session: Session, player_id: int) -> dict:
    """Get prediction accuracy for a specific player."""
    results = (
        session.query(PredictionResult)
        .join(Prediction)
        .filter(Prediction.player_id == player_id)
        .order_by(desc(PredictionResult.evaluated_at))
        .all()
    )

    if not results:
        return {"player_id": player_id, "total_evaluated": 0, "predictions": []}

    predictions = []
    for r in results:
        pred = r.prediction
        predictions.append({
            "prediction_id": pred.id,
            "date": pred.created_at.isoformat() if pred.created_at else None,
            "predicted": {
                "hits": pred.predicted_hits,
                "hr": pred.predicted_hr,
                "rbi": pred.predicted_rbi,
                "walks": pred.predicted_walks,
            },
            "actual": {
                "hits": r.actual_hits,
                "hr": r.actual_hr,
                "rbi": r.actual_rbi,
                "walks": r.actual_walks,
            },
            "mse": r.mse,
        })

    avg_mse = sum(r.mse for r in results) / len(results)

    return {
        "player_id": player_id,
        "total_evaluated": len(results),
        "avg_mse": round(avg_mse, 4),
        "predictions": predictions,
    }


def get_calibration_data(session: Session, n_bins: int = 10) -> list[dict]:
    """Generate calibration curve data.

    Groups predictions by confidence level and computes
    actual accuracy in each bin.
    """
    results = (
        session.query(PredictionResult, Prediction)
        .join(Prediction)
        .filter(Prediction.confidence.isnot(None))
        .all()
    )

    if not results:
        return []

    # Bin by confidence
    bins = [[] for _ in range(n_bins)]
    for r, pred in results:
        conf = pred.confidence or 0.5
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        # "Accurate" if avg absolute error < 1 across all stats
        avg_err = (
            abs((pred.predicted_hits or 0) - (r.actual_hits or 0))
            + abs((pred.predicted_hr or 0) - (r.actual_hr or 0))
            + abs((pred.predicted_rbi or 0) - (r.actual_rbi or 0))
            + abs((pred.predicted_walks or 0) - (r.actual_walks or 0))
        ) / 4.0
        bins[bin_idx].append(avg_err < 1.0)

    calibration = []
    for i, bin_data in enumerate(bins):
        bin_center = (i + 0.5) / n_bins
        if bin_data:
            actual_accuracy = sum(bin_data) / len(bin_data)
        else:
            actual_accuracy = None
        calibration.append({
            "confidence_bin": round(bin_center, 2),
            "predicted_accuracy": round(bin_center, 2),
            "actual_accuracy": round(actual_accuracy, 4) if actual_accuracy is not None else None,
            "n_predictions": len(bin_data),
        })

    return calibration
