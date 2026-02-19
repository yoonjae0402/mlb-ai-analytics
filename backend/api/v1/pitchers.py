"""Pitcher search and stats endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.api.v1.schemas import PlayerResponse
from backend.db.session import get_db
from backend.db.models import Player, PlayerStat

router = APIRouter(prefix="/pitchers", tags=["pitchers"])


@router.get("/search", response_model=list[PlayerResponse])
async def search_pitchers(
    q: str = "",
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Search for pitchers by name."""
    query = select(Player).where(Player.position == "P")
    if q:
        query = query.where(Player.name.ilike(f"%{q}%"))
    query = query.order_by(Player.name).limit(limit)
    result = await db.execute(query)
    players = result.scalars().all()
    return [
        PlayerResponse(
            id=p.id, mlb_id=p.mlb_id, name=p.name, team=p.team,
            position=p.position, bats=p.bats, throws=p.throws,
            headshot_url=p.headshot_url, current_level=p.current_level,
        )
        for p in players
    ]


@router.get("/{pitcher_id}/stats", response_model=dict)
async def pitcher_stats(
    pitcher_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get pitcher profile with computed ERA, WHIP, K/9, BB/9 and season totals.

    Response schema:
        {
            id, mlb_id, name, team, throws, headshot_url,
            stats: {
                total_games, total_innings, total_strikeouts,
                total_walks, total_earned_runs,
                era, whip, k_per_9, bb_per_9
            }
        }
    """
    result = await db.execute(select(Player).where(Player.id == pitcher_id))
    player = result.scalar_one_or_none()
    if not player:
        raise HTTPException(404, "Pitcher not found")

    # Aggregate pitching stats from player_stats rows
    agg_result = await db.execute(
        select(
            func.count(PlayerStat.id).label("total_games"),
            func.coalesce(func.sum(PlayerStat.innings_pitched), 0.0).label("total_innings"),
            func.coalesce(func.sum(PlayerStat.strikeouts), 0).label("total_strikeouts"),
            func.coalesce(func.sum(PlayerStat.walks), 0).label("total_walks"),
            func.coalesce(func.sum(PlayerStat.earned_runs), 0).label("total_earned_runs"),
            func.coalesce(func.sum(PlayerStat.hits), 0).label("total_hits_allowed"),
        )
        .where(PlayerStat.player_id == pitcher_id)
    )
    agg = agg_result.one_or_none()

    total_games = agg.total_games if agg else 0
    total_innings = float(agg.total_innings or 0) if agg else 0.0
    total_strikeouts = int(agg.total_strikeouts or 0) if agg else 0
    total_walks = int(agg.total_walks or 0) if agg else 0
    total_earned_runs = int(agg.total_earned_runs or 0) if agg else 0
    total_hits_allowed = int(agg.total_hits_allowed or 0) if agg else 0

    # Compute rate stats — guard against division by zero
    era = round(total_earned_runs / total_innings * 9, 2) if total_innings > 0 else None
    whip = round((total_walks + total_hits_allowed) / total_innings, 3) if total_innings > 0 else None
    k_per_9 = round(total_strikeouts / total_innings * 9, 1) if total_innings > 0 else None
    bb_per_9 = round(total_walks / total_innings * 9, 1) if total_innings > 0 else None

    return {
        "id": player.id,
        "mlb_id": player.mlb_id,
        "name": player.name,
        "team": player.team,
        "throws": player.throws,
        "headshot_url": player.headshot_url,
        "stats": {
            "total_games": total_games,
            "total_innings": round(total_innings, 1),
            "total_strikeouts": total_strikeouts,
            "total_walks": total_walks,
            "total_earned_runs": total_earned_runs,
            "era": era,
            "whip": whip,
            "k_per_9": k_per_9,
            "bb_per_9": bb_per_9,
        },
    }
