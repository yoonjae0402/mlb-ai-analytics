"""Schedule and calendar endpoints."""

from datetime import date, timedelta
from fastapi import APIRouter, Query

from backend.api.v1.schemas import ScheduleResponse, ScheduleGameResponse
from src.data.pipeline import MLBDataPipeline

router = APIRouter(prefix="/schedule", tags=["schedule"])

_pipeline = MLBDataPipeline()


@router.get("/range", response_model=ScheduleResponse)
async def schedule_range(
    start_date: str = Query(None, description="YYYY-MM-DD"),
    end_date: str = Query(None, description="YYYY-MM-DD"),
):
    """Get games for a date range (calendar view)."""
    if not start_date:
        start_date = date.today().strftime("%Y-%m-%d")
    if not end_date:
        end_date = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")

    raw_games = _pipeline.fetch_schedule_range(start_date, end_date)

    games = []
    for g in raw_games:
        games.append(ScheduleGameResponse(
            game_id=g.get("game_id", 0),
            game_date=g.get("game_date", start_date),
            away_team=g.get("away_name", g.get("away", "")),
            home_team=g.get("home_name", g.get("home", "")),
            away_score=g.get("away_score"),
            home_score=g.get("home_score"),
            status=g.get("status", "Scheduled"),
            venue=g.get("venue_name", g.get("venue", "")),
            away_probable_pitcher=g.get("away_probable_pitcher", "TBD"),
            home_probable_pitcher=g.get("home_probable_pitcher", "TBD"),
            game_datetime=g.get("game_datetime"),
            home_win_prob=None,
        ))

    return ScheduleResponse(
        games=games,
        start_date=start_date,
        end_date=end_date,
    )


@router.get("/today", response_model=ScheduleResponse)
async def schedule_today():
    """Shortcut for today's schedule."""
    today = date.today().strftime("%Y-%m-%d")
    raw_games = _pipeline.fetch_schedule(today)

    games = []
    for g in raw_games:
        games.append(ScheduleGameResponse(
            game_id=g.get("game_id", 0),
            game_date=today,
            away_team=g.get("away_name", g.get("away", "")),
            home_team=g.get("home_name", g.get("home", "")),
            away_score=g.get("away_score"),
            home_score=g.get("home_score"),
            status=g.get("status", "Scheduled"),
            venue=g.get("venue_name", g.get("venue", "")),
            away_probable_pitcher=g.get("away_probable_pitcher", "TBD"),
            home_probable_pitcher=g.get("home_probable_pitcher", "TBD"),
            game_datetime=g.get("game_datetime"),
            home_win_prob=None,
        ))

    return ScheduleResponse(
        games=games,
        start_date=today,
        end_date=today,
    )
