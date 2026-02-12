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
async def leaderboard(limit: int = 25):
    """Top predicted performers based on latest predictions."""
    from backend.db.session import SyncSessionLocal
    from backend.db.models import Prediction, Player
    from sqlalchemy import desc

    session = SyncSessionLocal()
    try:
        # Get latest prediction per player, ranked by predicted hits + HR + RBI
        from sqlalchemy import func
        subq = (
            session.query(
                Prediction.player_id,
                func.max(Prediction.created_at).label("latest"),
            )
            .group_by(Prediction.player_id)
            .subquery()
        )
        preds = (
            session.query(Prediction, Player)
            .join(Player, Prediction.player_id == Player.id)
            .join(
                subq,
                (Prediction.player_id == subq.c.player_id)
                & (Prediction.created_at == subq.c.latest),
            )
            .order_by(
                desc(Prediction.predicted_hits + Prediction.predicted_hr + Prediction.predicted_rbi)
            )
            .limit(limit)
            .all()
        )

        return [
            {
                "rank": i + 1,
                "player_id": pred.player_id,
                "player_name": player.name,
                "team": player.team,
                "headshot_url": player.headshot_url,
                "predicted_hits": pred.predicted_hits,
                "predicted_hr": pred.predicted_hr,
                "predicted_rbi": pred.predicted_rbi,
                "predicted_walks": pred.predicted_walks,
                "confidence": pred.confidence,
                "composite_score": round(
                    (pred.predicted_hits or 0)
                    + (pred.predicted_hr or 0) * 4
                    + (pred.predicted_rbi or 0),
                    3,
                ),
            }
            for i, (pred, player) in enumerate(preds)
        ]
    finally:
        session.close()
