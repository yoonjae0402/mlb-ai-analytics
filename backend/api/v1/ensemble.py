"""Ensemble prediction endpoints."""

from fastapi import APIRouter, HTTPException

from backend.api.dependencies import get_model_service
from backend.api.v1.schemas import (
    EnsembleRequest, EnsembleResponse, WeightSensitivityResponse
)
from backend.db.session import SyncSessionLocal

router = APIRouter(prefix="/ensemble", tags=["ensemble"])


@router.post("/predict", response_model=EnsembleResponse)
async def ensemble_predict(request: EnsembleRequest):
    """Ensemble prediction with configurable strategy and weights."""
    svc = get_model_service()
    session = SyncSessionLocal()
    try:
        return svc.get_ensemble_prediction(
            session,
            request.player_id,
            request.strategy,
            request.weights,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        session.close()


@router.get("/weight-sensitivity", response_model=WeightSensitivityResponse)
async def weight_sensitivity():
    """Sweep ensemble weights from 0 to 1 and return MSE curve."""
    svc = get_model_service()
    try:
        return svc.get_weight_sensitivity()
    except ValueError as e:
        raise HTTPException(400, str(e))
