"""Prediction accuracy tracking endpoints."""

import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.api.v1.schemas import (
    AccuracySummaryResponse,
    PlayerAccuracyResponse,
    CalibrationPoint,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/accuracy", tags=["accuracy"])


@router.get("/summary", response_model=AccuracySummaryResponse)
async def accuracy_summary():
    """Overall model accuracy metrics across all evaluated predictions."""
    from backend.db.session import SyncSessionLocal
    from backend.tasks.backfill_results import get_accuracy_summary

    session = SyncSessionLocal()
    try:
        result = get_accuracy_summary(session)
        return AccuracySummaryResponse(**result)
    finally:
        session.close()


@router.get("/by-player/{player_id}", response_model=PlayerAccuracyResponse)
async def accuracy_by_player(player_id: int):
    """Per-player prediction track record."""
    from backend.db.session import SyncSessionLocal
    from backend.tasks.backfill_results import get_accuracy_by_player

    session = SyncSessionLocal()
    try:
        result = get_accuracy_by_player(session, player_id)
        return PlayerAccuracyResponse(**result)
    finally:
        session.close()


@router.get("/calibration", response_model=list[CalibrationPoint])
async def calibration_data():
    """Calibration curve data — predicted vs actual accuracy by confidence."""
    from backend.db.session import SyncSessionLocal
    from backend.tasks.backfill_results import get_calibration_data

    session = SyncSessionLocal()
    try:
        return get_calibration_data(session)
    finally:
        session.close()


@router.post("/backfill")
async def trigger_backfill(background_tasks: BackgroundTasks, lookback_days: int = 7):
    """Trigger backfill of prediction results from completed games."""
    from backend.db.session import SyncSessionLocal
    from backend.tasks.backfill_results import backfill_prediction_results

    def run_backfill():
        session = SyncSessionLocal()
        try:
            count = backfill_prediction_results(session, lookback_days)
            logger.info(f"Backfill complete: {count} results created.")
        finally:
            session.close()

    background_tasks.add_task(run_backfill)
    return {"status": "backfill_started", "lookback_days": lookback_days}
