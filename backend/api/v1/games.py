"""Live games and schedule endpoints."""

from fastapi import APIRouter

from backend.api.v1.schemas import LiveGamesResponse, GameResponse
from src.services.realtime import fetch_live_games

router = APIRouter(prefix="/games", tags=["games"])


@router.get("/live", response_model=LiveGamesResponse)
async def live_games():
    """Get live or scheduled games from MLB Stats API."""
    games, mode = fetch_live_games()
    return LiveGamesResponse(
        games=[GameResponse(**g) for g in games],
        mode=mode,
    )


@router.get("/today", response_model=LiveGamesResponse)
async def todays_games():
    """Get today's schedule with predictions."""
    games, mode = fetch_live_games()
    return LiveGamesResponse(
        games=[GameResponse(**g) for g in games],
        mode=mode,
    )
