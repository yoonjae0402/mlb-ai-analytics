"""Model evaluation and tuning endpoints."""

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.api.dependencies import get_model_service
from backend.api.v1.schemas import EvaluationResponse, TuneRequest, TuneStatus

router = APIRouter(tags=["evaluation"])


@router.get("/model/metrics", response_model=dict)
async def model_metrics():
    """Model performance metrics from training."""
    svc = get_model_service()
    result = {}
    if svc.lstm_model:
        result["lstm"] = svc.lstm_model.metrics
    if svc.xgboost_model:
        result["xgboost"] = svc.xgboost_model.metrics
    return result


@router.get("/model/evaluation", response_model=EvaluationResponse)
async def model_evaluation():
    """Full evaluation report with baselines."""
    svc = get_model_service()
    if not svc.is_trained:
        raise HTTPException(400, "No trained models")

    try:
        from backend.core.evaluation import run_evaluation
        return run_evaluation(svc)
    except ImportError:
        # If evaluation module not yet created, return basic metrics
        result = {}
        if svc.lstm_model:
            result["lstm"] = svc.lstm_model.metrics
        if svc.xgboost_model:
            result["xgboost"] = svc.xgboost_model.metrics
        return EvaluationResponse(**result)


@router.get("/leaderboard", response_model=list[dict])
async def leaderboard():
    """Top predicted performers."""
    svc = get_model_service()
    # Placeholder — populated once predictions exist
    return []
