"""Predictions hub — central place for daily model outputs."""

from datetime import datetime, date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from backend.db.session import get_db
from backend.db.models import Player, Prediction
from backend.api.v1.schemas import (
    DailyPredictionResponse, PredictionsHubResponse,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/daily", response_model=PredictionsHubResponse)
async def daily_predictions(
    limit: int = Query(50, ge=1, le=200),
    sort_by: str = Query("predicted_hr", description="Sort field: predicted_hits, predicted_hr, predicted_rbi"),
    db: AsyncSession = Depends(get_db),
):
    """Get today's (or latest) predictions sorted by a stat.

    Returns the most recent batch of predictions — these are generated
    automatically by the scheduler or manually via the train pipeline.
    """
    # Find the latest prediction timestamp
    latest_ts = (await db.execute(
        select(func.max(Prediction.created_at))
    )).scalar()

    if not latest_ts:
        return PredictionsHubResponse(predictions=[], total=0, last_updated=None)

    # Get predictions from that batch (within 1 hour of latest)
    cutoff = latest_ts - timedelta(hours=1)

    # Determine sort column
    sort_col = {
        "predicted_hits": Prediction.predicted_hits,
        "predicted_hr": Prediction.predicted_hr,
        "predicted_rbi": Prediction.predicted_rbi,
        "predicted_walks": Prediction.predicted_walks,
    }.get(sort_by, Prediction.predicted_hr)

    result = await db.execute(
        select(Prediction, Player)
        .join(Player, Prediction.player_id == Player.id)
        .where(Prediction.created_at >= cutoff)
        .order_by(desc(sort_col))
        .limit(limit)
    )
    rows = result.all()

    predictions = []
    for pred, player in rows:
        predictions.append(DailyPredictionResponse(
            player_id=player.id,
            player_name=player.name,
            team=player.team,
            headshot_url=player.headshot_url,
            predicted_hits=pred.predicted_hits or 0,
            predicted_hr=pred.predicted_hr or 0,
            predicted_rbi=pred.predicted_rbi or 0,
            predicted_walks=pred.predicted_walks or 0,
            confidence=pred.confidence,
            created_at=pred.created_at.isoformat() if pred.created_at else None,
        ))

    return PredictionsHubResponse(
        predictions=predictions,
        total=len(predictions),
        last_updated=latest_ts.isoformat() if latest_ts else None,
    )


@router.get("/best-bets", response_model=PredictionsHubResponse)
async def best_bets(
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    """Top high-confidence predictions for today's games.

    Returns predictions where the model has highest confidence
    (highest combined predicted output), intended as the
    'Benefit of the Doubt' widget.
    """
    latest_ts = (await db.execute(
        select(func.max(Prediction.created_at))
    )).scalar()

    if not latest_ts:
        return PredictionsHubResponse(predictions=[], total=0, last_updated=None)

    cutoff = latest_ts - timedelta(hours=1)

    # Score = predicted_hits + predicted_hr*4 + predicted_rbi*2 + predicted_walks
    # This weights HRs and RBIs higher as they're rarer and more impactful
    score_expr = (
        Prediction.predicted_hits
        + Prediction.predicted_hr * 4
        + Prediction.predicted_rbi * 2
        + Prediction.predicted_walks
    )

    result = await db.execute(
        select(Prediction, Player)
        .join(Player, Prediction.player_id == Player.id)
        .where(Prediction.created_at >= cutoff)
        .order_by(desc(score_expr))
        .limit(limit)
    )
    rows = result.all()

    predictions = []
    for pred, player in rows:
        predictions.append(DailyPredictionResponse(
            player_id=player.id,
            player_name=player.name,
            team=player.team,
            headshot_url=player.headshot_url,
            predicted_hits=pred.predicted_hits or 0,
            predicted_hr=pred.predicted_hr or 0,
            predicted_rbi=pred.predicted_rbi or 0,
            predicted_walks=pred.predicted_walks or 0,
            confidence=pred.confidence,
            created_at=pred.created_at.isoformat() if pred.created_at else None,
        ))

    return PredictionsHubResponse(
        predictions=predictions,
        total=len(predictions),
        last_updated=latest_ts.isoformat() if latest_ts else None,
    )
