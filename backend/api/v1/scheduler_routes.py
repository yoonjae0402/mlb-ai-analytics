"""Scheduler status and control endpoints."""

from fastapi import APIRouter

from backend.api.v1.schemas import SchedulerStatus
from src.services.scheduler import get_scheduler

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


@router.get("/status", response_model=SchedulerStatus)
async def scheduler_status():
    """Get current scheduler status."""
    sched = get_scheduler()
    return SchedulerStatus(**sched.get_status())


@router.post("/run")
async def trigger_run():
    """Manually trigger the prediction pipeline."""
    sched = get_scheduler()
    sched.run_now()
    return {"status": "started", "message": "Prediction pipeline triggered"}
