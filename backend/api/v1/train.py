"""Model training endpoints."""

from fastapi import APIRouter, BackgroundTasks

from backend.api.dependencies import get_model_service
from backend.api.v1.schemas import TrainRequest, TrainStatus, TrainCurves
from backend.db.session import SyncSessionLocal
from backend.db.models import ModelVersion

router = APIRouter(prefix="/train", tags=["training"])


def _run_training(config: dict):
    """Background task for training."""
    svc = get_model_service()
    session = SyncSessionLocal()
    try:
        results = svc.train_models(session, config)

        # Store model versions in DB
        for model_type in ["lstm", "xgboost"]:
            if model_type in results:
                r = results[model_type]
                mv = ModelVersion(
                    model_type=model_type,
                    version=r["version"],
                    hyperparams_json=config,
                    train_mse=r["metrics"].get("final_train_loss"),
                    val_mse=r["metrics"].get("mse"),
                    train_r2=None,
                    val_r2=r["metrics"].get("r2"),
                    checkpoint_path=r["checkpoint"],
                )
                session.add(mv)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("", response_model=dict)
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
):
    """Start model training in the background."""
    svc = get_model_service()
    if svc.training_status["is_training"]:
        return {"status": "already_training", "message": "Training is already in progress"}

    config = request.model_dump()
    background_tasks.add_task(_run_training, config)
    return {"status": "started", "config": config}


@router.get("/status", response_model=TrainStatus)
async def training_status():
    """Get current training status."""
    svc = get_model_service()
    return svc.get_training_status()


@router.get("/curves", response_model=TrainCurves)
async def training_curves():
    """Get training loss curves."""
    svc = get_model_service()
    return svc.get_training_curves()
