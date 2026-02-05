"""Hyperparameter tuning endpoints."""

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.api.dependencies import get_model_service
from backend.api.v1.schemas import TuneRequest, TuneStatus
from backend.core.tuning import tune_lstm, tune_xgboost, get_tuning_status

router = APIRouter(prefix="/tune", tags=["tuning"])


def _run_tuning(model_type: str, n_trials: int):
    svc = get_model_service()
    if svc.X_train is None:
        return
    if model_type == "lstm":
        tune_lstm(svc.X_train, svc.y_train, svc.X_val, svc.y_val, n_trials)
    else:
        tune_xgboost(svc.X_train, svc.y_train, svc.X_val, svc.y_val, n_trials)


@router.post("")
async def start_tuning(request: TuneRequest, background_tasks: BackgroundTasks):
    """Trigger Optuna hyperparameter tuning in the background."""
    svc = get_model_service()
    if not svc.is_trained:
        raise HTTPException(400, "Train models first before tuning")

    status = get_tuning_status()
    if status["is_tuning"]:
        return {"status": "already_tuning"}

    background_tasks.add_task(_run_tuning, request.model_type, request.n_trials)
    return {"status": "started", "model_type": request.model_type, "n_trials": request.n_trials}


@router.get("/status", response_model=TuneStatus)
async def tuning_status():
    """Get tuning progress and best params."""
    return get_tuning_status()
