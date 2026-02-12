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
    """Get pitcher profile with aggregated stats."""
    result = await db.execute(select(Player).where(Player.id == pitcher_id))
    player = result.scalar_one_or_none()
    if not player:
        raise HTTPException(404, "Pitcher not found")

    # Get recent game stats
    stats_q = await db.execute(
        select(PlayerStat)
        .where(PlayerStat.player_id == pitcher_id)
        .order_by(PlayerStat.game_date.desc())
        .limit(30)
    )
    stats = stats_q.scalars().all()

    # Aggregate
    games_pitched = len(stats)
    total_k = sum(s.strikeouts or 0 for s in stats)
    total_bb = sum(s.walks or 0 for s in stats)
    total_hits_allowed = sum(s.hits or 0 for s in stats)

    return {
        "player": {
            "id": player.id,
            "mlb_id": player.mlb_id,
            "name": player.name,
            "team": player.team,
            "throws": player.throws,
            "headshot_url": player.headshot_url,
        },
        "games_pitched": games_pitched,
        "total_strikeouts": total_k,
        "total_walks": total_bb,
        "total_hits_allowed": total_hits_allowed,
        "k_per_game": round(total_k / max(games_pitched, 1), 2),
        "bb_per_game": round(total_bb / max(games_pitched, 1), 2),
    }
