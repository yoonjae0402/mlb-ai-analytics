"""Player search and profile endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.db.session import get_db
from backend.db.models import Player, PlayerStat, Prediction
from backend.api.v1.schemas import PlayerResponse, PlayerDetail, PlayerStatsResponse

router = APIRouter(prefix="/players", tags=["players"])


@router.get("/search", response_model=list[PlayerResponse])
async def search_players(
    q: str = Query(..., min_length=2),
    limit: int = Query(10, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Search players by name."""
    result = await db.execute(
        select(Player)
        .where(Player.name.ilike(f"%{q}%"))
        .order_by(Player.name)
        .limit(limit)
    )
    players = result.scalars().all()
    return [
        PlayerResponse(
            id=p.id, mlb_id=p.mlb_id, name=p.name,
            team=p.team, position=p.position, bats=p.bats, throws=p.throws,
        )
        for p in players
    ]


@router.get("/{player_id}", response_model=PlayerDetail)
async def get_player(player_id: int, db: AsyncSession = Depends(get_db)):
    """Get player profile with recent stats."""
    result = await db.execute(select(Player).where(Player.id == player_id))
    player = result.scalar_one_or_none()
    if not player:
        raise HTTPException(404, "Player not found")

    # Recent stats (last 20 games)
    stats_result = await db.execute(
        select(PlayerStat)
        .where(PlayerStat.player_id == player_id)
        .order_by(PlayerStat.game_date.desc())
        .limit(20)
    )
    stats = stats_result.scalars().all()

    # Season totals
    totals_result = await db.execute(
        select(
            func.sum(PlayerStat.hits).label("hits"),
            func.sum(PlayerStat.home_runs).label("home_runs"),
            func.sum(PlayerStat.rbi).label("rbi"),
            func.sum(PlayerStat.walks).label("walks"),
            func.sum(PlayerStat.at_bats).label("at_bats"),
            func.count(PlayerStat.id).label("games"),
        )
        .where(PlayerStat.player_id == player_id)
    )
    totals = totals_result.one_or_none()

    season_totals = None
    if totals and totals.games:
        ab = totals.at_bats or 1
        season_totals = {
            "games": totals.games,
            "hits": totals.hits or 0,
            "home_runs": totals.home_runs or 0,
            "rbi": totals.rbi or 0,
            "walks": totals.walks or 0,
            "at_bats": totals.at_bats or 0,
            "batting_avg": round((totals.hits or 0) / ab, 3),
        }

    return PlayerDetail(
        player=PlayerResponse(
            id=player.id, mlb_id=player.mlb_id, name=player.name,
            team=player.team, position=player.position,
            bats=player.bats, throws=player.throws,
        ),
        recent_stats=[
            PlayerStatsResponse(
                game_date=s.game_date,
                batting_avg=s.batting_avg, obp=s.obp, slg=s.slg,
                hits=s.hits, home_runs=s.home_runs,
                rbi=s.rbi, walks=s.walks,
            )
            for s in stats
        ],
        season_totals=season_totals,
    )


@router.get("/{player_id}/predictions", response_model=list[dict])
async def get_player_predictions(
    player_id: int, db: AsyncSession = Depends(get_db)
):
    """Get historical predictions for a player."""
    result = await db.execute(
        select(Prediction)
        .where(Prediction.player_id == player_id)
        .order_by(Prediction.created_at.desc())
        .limit(50)
    )
    predictions = result.scalars().all()

    return [
        {
            "id": p.id,
            "predicted_hits": p.predicted_hits,
            "predicted_hr": p.predicted_hr,
            "predicted_rbi": p.predicted_rbi,
            "predicted_walks": p.predicted_walks,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in predictions
    ]
