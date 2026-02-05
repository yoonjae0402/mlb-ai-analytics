"""Background data refresh task.

Daily job to fetch new game results, update player_stats,
and evaluate previous predictions against actual results.
"""

import logging
from datetime import date, timedelta

from backend.db.session import SyncSessionLocal
from backend.db.models import Player, PlayerStat, Prediction, PredictionResult, Game
from src.data.pipeline import MLBDataPipeline

logger = logging.getLogger(__name__)


def refresh_daily_data():
    """Fetch yesterday's game results and update the database."""
    pipeline = MLBDataPipeline()
    session = SyncSessionLocal()
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        # Fetch yesterday's schedule
        games = pipeline.fetch_schedule(yesterday)
        logger.info(f"Found {len(games)} games from {yesterday}")

        for g in games:
            game_id = g.get("game_id")
            if not game_id:
                continue

            # Upsert game
            existing = session.query(Game).filter_by(mlb_game_id=game_id).first()
            if existing is None:
                game_obj = Game(
                    mlb_game_id=game_id,
                    game_date=date.fromisoformat(yesterday),
                    away_team=g.get("away_name", ""),
                    home_team=g.get("home_name", ""),
                    away_score=g.get("away_score"),
                    home_score=g.get("home_score"),
                    status=g.get("status", ""),
                    venue=g.get("venue_name", ""),
                )
                session.add(game_obj)

        session.commit()

        # Evaluate predictions
        _evaluate_predictions(session, yesterday)
        session.commit()

        logger.info("Daily data refresh complete")

    except Exception as e:
        session.rollback()
        logger.error(f"Error during daily refresh: {e}")
        raise
    finally:
        session.close()


def _evaluate_predictions(session, game_date: str):
    """Compare predictions to actual results for a given date."""
    target_date = date.fromisoformat(game_date)

    # Find predictions for this date that haven't been evaluated
    predictions = (
        session.query(Prediction)
        .filter(Prediction.created_at >= target_date.isoformat())
        .filter(~Prediction.result.has())
        .all()
    )

    evaluated = 0
    for pred in predictions:
        # Find the actual stats for this player on this date
        actual = (
            session.query(PlayerStat)
            .filter_by(player_id=pred.player_id, game_date=target_date)
            .first()
        )
        if actual is None:
            continue

        # Compute MSE
        diffs = [
            (pred.predicted_hits - (actual.hits or 0)) ** 2,
            (pred.predicted_hr - (actual.home_runs or 0)) ** 2,
            (pred.predicted_rbi - (actual.rbi or 0)) ** 2,
            (pred.predicted_walks - (actual.walks or 0)) ** 2,
        ]
        mse = sum(diffs) / len(diffs)

        result = PredictionResult(
            prediction_id=pred.id,
            actual_hits=actual.hits or 0,
            actual_hr=actual.home_runs or 0,
            actual_rbi=actual.rbi or 0,
            actual_walks=actual.walks or 0,
            mse=mse,
        )
        session.add(result)
        evaluated += 1

    logger.info(f"Evaluated {evaluated} predictions for {game_date}")
