"""Attention visualization endpoints."""

from fastapi import APIRouter, HTTPException

from backend.api.dependencies import get_model_service
from backend.api.v1.schemas import (
    AttentionRequest, AttentionResponse, FeatureAttributionResponse
)

router = APIRouter(prefix="/attention", tags=["attention"])


@router.post("/weights", response_model=AttentionResponse)
async def attention_weights(request: AttentionRequest):
    """Get attention heatmap for a validation sample."""
    svc = get_model_service()
    try:
        return svc.get_attention_weights(request.sample_idx)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/feature-attribution", response_model=FeatureAttributionResponse)
async def feature_attribution(request: AttentionRequest):
    """Get gradient-based feature importance."""
    svc = get_model_service()
    try:
        return svc.get_feature_attribution(request.sample_idx)
    except ValueError as e:
        raise HTTPException(400, str(e))
