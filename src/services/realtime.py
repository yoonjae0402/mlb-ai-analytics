"""
Real-time game data service.

Polls MLB Stats API for live games, computes win probability,
and falls back to synthetic demo data when no live games are available.
"""

import logging
from typing import Optional

import numpy as np

from src.data.synthetic import generate_live_game_data

logger = logging.getLogger(__name__)


def fetch_live_games() -> tuple[list[dict], str]:
    """
    Fetch live game data. Returns (games, mode).

    mode is one of:
        "live"  - real games from MLB Stats API
        "schedule" - no live games, showing today's schedule
        "demo" - API unavailable, using synthetic data
    """
    try:
        import statsapi
        live = statsapi.get("game", {})
        # Try fetching today's scoreboard
        scoreboard = statsapi.schedule()

        live_games = [g for g in scoreboard if g.get("status", "") in ("In Progress", "Live")]

        if live_games:
            games = [_parse_live_game(g) for g in live_games]
            return games, "live"

        if scoreboard:
            games = [_parse_scheduled_game(g) for g in scoreboard[:8]]
            return games, "schedule"

    except Exception as e:
        logger.debug(f"MLB Stats API unavailable: {e}")

    # Fallback to demo mode
    games = generate_live_game_data(n_games=6)
    return games, "demo"


def advance_inning(game: dict) -> dict:
    """
    Simulate advancing a game by one half-inning.
    Updates score, inning, and win probability.
    """
    rng = np.random.RandomState()

    game = dict(game)
    wp_history = list(game.get("wp_history", [0.5]))

    if game["half"] == "Top":
        # Away team bats
        runs = int(rng.choice([0, 0, 0, 0, 0, 1, 1, 2, 3], p=[0.55, 0.10, 0.08, 0.05, 0.02, 0.10, 0.05, 0.03, 0.02]))
        game["away_score"] += runs
        game["half"] = "Bottom"
    else:
        # Home team bats
        runs = int(rng.choice([0, 0, 0, 0, 0, 1, 1, 2, 3], p=[0.50, 0.10, 0.08, 0.05, 0.02, 0.12, 0.06, 0.04, 0.03]))
        game["home_score"] += runs
        game["half"] = "Top"
        game["inning"] += 1

    if game["inning"] > 9 and game["home_score"] != game["away_score"]:
        game["status"] = "Final"

    # Recompute win probability
    score_diff = game["home_score"] - game["away_score"]
    leverage = min(game["inning"] / 9.0, 1.0)
    home_wp = 1.0 / (1.0 + np.exp(-0.6 * score_diff * (0.5 + 0.5 * leverage)))
    home_wp = float(np.clip(home_wp + rng.randn() * 0.03, 0.02, 0.98))

    if game["status"] == "Final":
        home_wp = 1.0 if game["home_score"] > game["away_score"] else 0.0

    game["home_win_prob"] = home_wp
    wp_history.append(home_wp)
    game["wp_history"] = wp_history

    return game


def compute_win_probability(
    home_score: int,
    away_score: int,
    inning: float,
    is_home_batting: bool = True,
) -> float:
    """
    Simple logistic win probability model.

    Based on score differential and game progress.
    """
    diff = home_score - away_score
    progress = min(inning / 9.0, 1.0)

    # Home team batting advantage
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
    }
