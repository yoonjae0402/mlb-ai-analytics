"""Real-time game data service.

Polls MLB Stats API for live games and computes win probability.
Returns real data only — no synthetic fallback.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def fetch_live_games() -> tuple[list[dict], str]:
    """Fetch live game data. Returns (games, mode).

    mode is one of:
        "live"     - real games currently in progress
        "schedule" - no live games, showing today's schedule
        "off_day"  - no games scheduled today
    """
    try:
        import statsapi
        scoreboard = statsapi.schedule()

        live_games = [g for g in scoreboard if g.get("status", "") in ("In Progress", "Live")]

        if live_games:
            games = [_parse_live_game(g) for g in live_games]
            return games, "live"

        if scoreboard:
            games = [_parse_scheduled_game(g) for g in scoreboard[:12]]
            return games, "schedule"

    except Exception as e:
        logger.warning(f"MLB Stats API unavailable: {e}")

    return [], "off_day"


def compute_win_probability(
    home_score: int,
    away_score: int,
    inning: float,
    is_home_batting: bool = True,
) -> float:
    """Simple logistic win probability model based on score differential and game progress."""
    diff = home_score - away_score
    progress = min(inning / 9.0, 1.0)

    if is_home_batting:
        diff += 0.1

    wp = 1.0 / (1.0 + np.exp(-0.5 * diff * (0.5 + 0.5 * progress)))
    return float(np.clip(wp, 0.01, 0.99))


def _parse_live_game(g: dict) -> dict:
    """Parse a live game from statsapi.schedule() format."""
    away_score = g.get("away_score", 0)
    home_score = g.get("home_score", 0)
    inning = g.get("current_inning", 1)

    return {
        "game_id": str(g.get("game_id", "")),
        "away_abbrev": g.get("away_name", "AWAY")[:3].upper(),
        "away_name": g.get("away_name", "Away Team"),
        "home_abbrev": g.get("home_name", "HOME")[:3].upper(),
        "home_name": g.get("home_name", "Home Team"),
        "away_score": away_score,
        "home_score": home_score,
        "inning": inning,
        "half": g.get("inning_state", "Top"),
        "status": g.get("status", "In Progress"),
        "home_win_prob": compute_win_probability(home_score, away_score, inning),
        "wp_history": [0.5],
    }


def _parse_scheduled_game(g: dict) -> dict:
    """Parse a scheduled (not yet started) game."""
    return {
        "game_id": str(g.get("game_id", "")),
        "away_abbrev": g.get("away_name", "AWAY")[:3].upper(),
        "away_name": g.get("away_name", "Away Team"),
        "home_abbrev": g.get("home_name", "HOME")[:3].upper(),
        "home_name": g.get("home_name", "Home Team"),
        "away_score": 0,
        "home_score": 0,
        "inning": 0,
        "half": "",
        "status": g.get("status", "Scheduled"),
        "home_win_prob": 0.5,
        "wp_history": [0.5],
        "home_probable_pitcher": g.get("home_probable_pitcher", "TBD"),
        "away_probable_pitcher": g.get("away_probable_pitcher", "TBD"),
        "game_datetime": g.get("game_datetime", ""),
        "venue": g.get("venue_name", ""),
    }
