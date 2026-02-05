"""Player prediction endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from backend.api.dependencies import get_model_service, get_sync_db
from backend.api.v1.schemas import PredictRequest, PredictionResponse
from backend.core.model_service import ModelService
from backend.db.models import Player, Prediction

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("/player", response_model=PredictionResponse)
async def predict_player(request: PredictRequest):
    """Predict next-game stats for a player."""
    svc = get_model_service()
    if not svc.is_trained:
        raise HTTPException(400, "No trained models. Train models first via POST /v1/train")

    from backend.db.session import SyncSessionLocal
    session = SyncSessionLocal()
    try:
        result = svc.predict_player(session, request.player_id, request.model_type)

        # Store prediction in DB
        # Find latest model version
        from backend.db.models import ModelVersion
        mv = (
            session.query(ModelVersion)
            .filter_by(model_type=request.model_type if request.model_type != "ensemble" else "lstm")
            .order_by(ModelVersion.trained_at.desc())
            .first()
        )
        if mv:
            pred = Prediction(
                player_id=request.player_id,
                model_version_id=mv.id,
                predicted_hits=result["predicted_hits"],
                predicted_hr=result["predicted_hr"],
                predicted_rbi=result["predicted_rbi"],
                predicted_walks=result["predicted_walks"],
            )
            session.add(pred)
            session.commit()

        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        session.close()
