"""System health endpoint — model status, DB stats, API info."""

from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/system/health")
async def system_health():
    """Full system health: model status, DB stats, uptime."""
    from backend.core.model_service import get_model_service
    from backend.db.session import SyncSessionLocal
    from backend.db.models import (
        Player, PlayerStat, Prediction, ModelVersion, Game
    )
    from sqlalchemy import desc
    import datetime

    svc = get_model_service()
    health = svc.get_health()

    # DB stats
    session = SyncSessionLocal()
    db_stats = {}
    try:
        db_stats["player_count"] = session.query(Player).count()
        db_stats["stat_rows"] = session.query(PlayerStat).count()
        db_stats["prediction_count"] = session.query(Prediction).count()
        db_stats["game_count"] = session.query(Game).count()

        # Latest model version timestamps per type
        model_versions = {}
        for model_type in ["lstm", "xgboost", "lightgbm", "linear"]:
            row = (
                session.query(ModelVersion)
                .filter_by(model_type=model_type)
                .order_by(desc(ModelVersion.trained_at))
                .first()
            )
            if row:
                model_versions[model_type] = {
                    "version": row.version,
                    "trained_at": row.trained_at.isoformat() if row.trained_at else None,
                    "val_mse": row.val_mse,
                    "val_r2": row.val_r2,
                    "checkpoint_path": row.checkpoint_path,
                }
        health["model_versions"] = model_versions
    finally:
        session.close()

    health["db"] = db_stats
    health["api_version"] = "2.0.0"
    health["timestamp"] = datetime.datetime.utcnow().isoformat()

    return health
