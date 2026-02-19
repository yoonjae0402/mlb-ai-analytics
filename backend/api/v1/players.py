"""Player search, index, profile, and comparison endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from backend.db.session import get_db
from backend.db.models import Player, PlayerStat, Prediction
from backend.api.v1.schemas import (
    PlayerResponse, PlayerDetail, PlayerStatsResponse, PlayerIndexResponse,
    PlayerCompareResponse,
)

router = APIRouter(prefix="/players", tags=["players"])


def _player_to_response(p: Player) -> PlayerResponse:
    return PlayerResponse(
        id=p.id, mlb_id=p.mlb_id, name=p.name,
        team=p.team, position=p.position, bats=p.bats, throws=p.throws,
        headshot_url=p.headshot_url,
        current_level=p.current_level,
        prospect_rank=p.prospect_rank,
        age=p.age,
    )


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
    return [_player_to_response(p) for p in players]


@router.get("/index", response_model=PlayerIndexResponse)
async def player_index(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=10, le=100),
    team: str = Query(None),
    level: str = Query(None),
    position: str = Query(None),
    search: str = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Paginated, filterable player index (MLB + MiLB)."""
    query = select(Player)
    count_query = select(func.count(Player.id))

    if team:
        query = query.where(Player.team == team)
        count_query = count_query.where(Player.team == team)
    if level:
        query = query.where(Player.current_level == level)
        count_query = count_query.where(Player.current_level == level)
    if position:
        query = query.where(Player.position == position)
        count_query = count_query.where(Player.position == position)
    if search:
        query = query.where(Player.name.ilike(f"%{search}%"))
        count_query = count_query.where(Player.name.ilike(f"%{search}%"))

    total = (await db.execute(count_query)).scalar() or 0

    offset = (page - 1) * per_page
    result = await db.execute(
        query.order_by(Player.name).offset(offset).limit(per_page)
    )
    players = result.scalars().all()

    return PlayerIndexResponse(
        players=[_player_to_response(p) for p in players],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/compare", response_model=PlayerCompareResponse)
async def compare_players(
    ids: str = Query(..., description="Comma-separated player IDs (max 4)"),
    db: AsyncSession = Depends(get_db),
):
    """Compare multiple players side-by-side."""
    try:
        player_ids = [int(x.strip()) for x in ids.split(",")]
    except ValueError:
        raise HTTPException(400, "Invalid player IDs format")

    if len(player_ids) > 4:
        raise HTTPException(400, "Maximum 4 players for comparison")
    if len(player_ids) < 2:
        raise HTTPException(400, "Need at least 2 players to compare")

    players_detail = []
    for pid in player_ids:
        result = await db.execute(select(Player).where(Player.id == pid))
        player = result.scalar_one_or_none()
        if not player:
            raise HTTPException(404, f"Player {pid} not found")

        stats_result = await db.execute(
            select(PlayerStat)
            .where(PlayerStat.player_id == pid)
            .order_by(PlayerStat.game_date.desc())
            .limit(20)
        )
        stats = stats_result.scalars().all()

        totals_result = await db.execute(
            select(
                func.sum(PlayerStat.hits).label("hits"),
                func.sum(PlayerStat.home_runs).label("home_runs"),
                func.sum(PlayerStat.rbi).label("rbi"),
                func.sum(PlayerStat.walks).label("walks"),
                func.sum(PlayerStat.at_bats).label("at_bats"),
                func.count(PlayerStat.id).label("games"),
            )
            .where(PlayerStat.player_id == pid)
        )
        totals = totals_result.one_or_none()

        season_totals = None
        if totals and totals.games:
            ab = totals.at_bats or 1
            h = totals.hits or 0
            bb = totals.walks or 0
            hr = totals.home_runs or 0
            # OBP = (H + BB) / (AB + BB)
            obp_denom = ab + bb if ab > 0 else 1
            obp = round((h + bb) / obp_denom, 3)
            # SLG approximation (we don't have 2B/3B, so use H + HR as a proxy for total bases)
            # Rough estimate: total_bases ≈ H + HR (assumes extra bases from HR)
            slg = round((h + hr) / ab, 3) if ab > 0 else 0.0

            season_totals = {
                "games": totals.games,
                "hits": h,
                "home_runs": hr,
                "rbi": totals.rbi or 0,
                "walks": bb,
                "at_bats": ab,
                "batting_avg": round(h / ab, 3),
                "obp": obp,
                "slg": slg,
            }

        players_detail.append(PlayerDetail(
            player=_player_to_response(player),
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
        ))

    return PlayerCompareResponse(players=players_detail)


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
        h = totals.hits or 0
        bb = totals.walks or 0
        hr = totals.home_runs or 0
        # OBP = (H + BB) / (AB + BB)
        obp_denom = ab + bb if ab > 0 else 1
        obp = round((h + bb) / obp_denom, 3)
        # SLG approximation (we don't have 2B/3B, so use H + HR as a proxy for total bases)
        # Rough estimate: total_bases ≈ H + HR (assumes extra bases from HR)
        slg = round((h + hr) / ab, 3) if ab > 0 else 0.0

        season_totals = {
            "games": totals.games,
            "hits": h,
            "home_runs": hr,
            "rbi": totals.rbi or 0,
            "walks": bb,
            "at_bats": ab,
            "batting_avg": round(h / ab, 3),
            "obp": obp,
            "slg": slg,
        }

    return PlayerDetail(
        player=_player_to_response(player),
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
