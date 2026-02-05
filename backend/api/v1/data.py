"""Data pipeline status and refresh endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.db.session import get_db
from backend.db.models import Player, PlayerStat, Game, Prediction
from backend.api.v1.schemas import DataStatus

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/status", response_model=DataStatus)
async def data_status(db: AsyncSession = Depends(get_db)):
    """Get data freshness and counts."""
    players_count = (await db.execute(select(func.count(Player.id)))).scalar() or 0
    stats_count = (await db.execute(select(func.count(PlayerStat.id)))).scalar() or 0
    games_count = (await db.execute(select(func.count(Game.id)))).scalar() or 0
    predictions_count = (await db.execute(select(func.count(Prediction.id)))).scalar() or 0

    # Last updated
    last_stat = (await db.execute(
        select(PlayerStat.game_date).order_by(PlayerStat.game_date.desc()).limit(1)
    )).scalar()

    # Distinct seasons
    seasons_result = await db.execute(
        select(func.extract("year", PlayerStat.game_date).label("yr"))
        .distinct()
        .order_by(func.extract("year", PlayerStat.game_date))
    )
    seasons = [int(r.yr) for r in seasons_result if r.yr]

    return DataStatus(
        players_count=players_count,
        stats_count=stats_count,
        games_count=games_count,
        predictions_count=predictions_count,
        last_updated=str(last_stat) if last_stat else None,
        seasons=seasons,
    )


@router.post("/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    """Trigger manual data refresh."""
    from backend.tasks.data_refresh import refresh_daily_data
    background_tasks.add_task(refresh_daily_data)
    return {"status": "started", "message": "Data refresh started in background"}
