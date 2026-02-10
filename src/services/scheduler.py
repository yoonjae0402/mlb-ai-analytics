"""Automated prediction scheduler.

Runs a daily cron job (default 4 AM ET) that:
1. Fetches the latest game data
2. Retrains / fine-tunes models on new data
3. Generates predictions for all upcoming games
4. Caches results in the DB for instant frontend access
"""

import logging
import threading
from datetime import datetime, date, timedelta
from typing import Optional

import schedule
import time

logger = logging.getLogger(__name__)


class PredictionScheduler:
    """Manages the automated daily prediction pipeline."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[str] = None
        self.status = "idle"  # idle, running, error
        self.last_error: Optional[str] = None

    def start(self, run_time: str = "04:00"):
        """Start the scheduler to run daily at the given time (ET).

        Args:
            run_time: Time in HH:MM format (24h, Eastern Time).
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        schedule.every().day.at(run_time).do(self._run_pipeline)
        self.next_run = run_time
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"Prediction scheduler started — runs daily at {run_time} ET")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        schedule.clear()
        logger.info("Prediction scheduler stopped")

    def run_now(self):
        """Trigger an immediate run (for manual/API use)."""
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def get_status(self) -> dict:
        """Return current scheduler status for the API."""
        return {
            "running": self._running,
            "status": self.status,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run,
            "last_error": self.last_error,
        }

    def _loop(self):
        """Background loop that checks for pending jobs."""
        while self._running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _run_pipeline(self):
        """Execute the full prediction pipeline."""
        self.status = "running"
        self.last_error = None
        logger.info("Starting automated prediction pipeline...")

        try:
            self._step_fetch_data()
            self._step_train_models()
            self._step_generate_predictions()
            self.status = "idle"
            self.last_run = datetime.utcnow()
            logger.info("Prediction pipeline completed successfully")
        except Exception as e:
            self.status = "error"
            self.last_error = str(e)
            logger.error(f"Prediction pipeline failed: {e}")

    def _step_fetch_data(self):
        """Step 1: Fetch latest game results and update the database."""
        logger.info("[Scheduler] Step 1: Fetching latest data...")
        from src.data.pipeline import MLBDataPipeline

        pipeline = MLBDataPipeline()
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        games = pipeline.fetch_schedule(yesterday)
        logger.info(f"[Scheduler] Fetched {len(games)} games from yesterday")

    def _step_train_models(self):
        """Step 2: Retrain models on the latest data."""
        logger.info("[Scheduler] Step 2: Retraining models...")
        from backend.db.session import SyncSessionLocal
        from backend.core.model_service import get_model_service

        service = get_model_service()
        if service.training_status.get("is_training"):
            logger.warning("[Scheduler] Training already in progress, skipping")
            return

        session = SyncSessionLocal()
        try:
            service.train_models(session, config={
                "epochs": 10,  # Quick fine-tune, not full training
                "lr": 0.0005,
                "hidden_size": 64,
                "batch_size": 32,
                "n_estimators": 200,
                "max_depth": 6,
                "xgb_lr": 0.1,
                "seasons": [2024],
            })
        finally:
            session.close()
        logger.info("[Scheduler] Model training complete")

    def _step_generate_predictions(self):
        """Step 3: Generate predictions for today's upcoming games."""
        logger.info("[Scheduler] Step 3: Generating predictions...")
        from src.data.pipeline import MLBDataPipeline
        from backend.db.session import SyncSessionLocal
        from backend.db.models import Player, Game, Prediction, ModelVersion
        from backend.core.model_service import get_model_service

        pipeline = MLBDataPipeline()
        service = get_model_service()
        today = date.today().strftime("%Y-%m-%d")
        games = pipeline.fetch_schedule(today)

        if not games:
            logger.info("[Scheduler] No games today, skipping predictions")
            return

        session = SyncSessionLocal()
        try:
            # Get the latest model version
            latest_model = (
                session.query(ModelVersion)
                .order_by(ModelVersion.trained_at.desc())
                .first()
            )
            if not latest_model:
                logger.warning("[Scheduler] No trained model available")
                return

            # Get all active MLB players
            players = session.query(Player).filter(
                Player.current_level == "MLB"
            ).all()

            prediction_count = 0
            for player in players:
                try:
                    result = service.predict_player(session, player.id, "ensemble")
                    if result:
                        pred = Prediction(
                            player_id=player.id,
                            model_version_id=latest_model.id,
                            predicted_hits=result.get("predicted_hits", 0),
                            predicted_hr=result.get("predicted_hr", 0),
                            predicted_rbi=result.get("predicted_rbi", 0),
                            predicted_walks=result.get("predicted_walks", 0),
                        )
                        session.add(pred)
                        prediction_count += 1
                except Exception as e:
                    logger.debug(f"Skipping prediction for player {player.id}: {e}")
                    continue

            session.commit()
            logger.info(f"[Scheduler] Generated {prediction_count} predictions")
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()


# Singleton instance
_scheduler: Optional[PredictionScheduler] = None


def get_scheduler() -> PredictionScheduler:
    """Get or create the singleton scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PredictionScheduler()
    return _scheduler
