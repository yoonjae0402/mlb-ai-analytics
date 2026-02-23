"""Current MLB standings from MLB Stats API."""

import time
import logging
from datetime import date
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/standings", tags=["standings"])

# Module-level cache (TTL: 10 minutes)
_cache: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 600  # 10 minutes


class TeamStanding(BaseModel):
    team_name: str
    team_abbreviation: str
    wins: int
    losses: int
    pct: float
    gb: str
    streak: str
    last_10: str
    home_record: Optional[str] = None
    away_record: Optional[str] = None
    division_rank: int
    wildcard_gb: Optional[str] = None


class DivisionStandings(BaseModel):
    division_name: str
    league: str
    teams: list[TeamStanding]


class StandingsResponse(BaseModel):
    divisions: list[DivisionStandings]
    as_of: str


# Full team name → abbreviation map
_ABBREVIATIONS: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    "Athletics": "OAK",
}

# Division name → league
_DIVISION_LEAGUES: dict[str, str] = {
    "AL East": "AL", "AL Central": "AL", "AL West": "AL",
    "NL East": "NL", "NL Central": "NL", "NL West": "NL",
}


def _format_streak(raw: str) -> str:
    """Normalize streak strings like 'W - 3', 'W3', 'L1' → 'W3', 'L1'."""
    if not raw:
        return "—"
    cleaned = raw.replace(" ", "").replace("-", "")
    return cleaned if cleaned else "—"


def _format_last10(raw: str) -> str:
    """Normalize last10 strings like '7 - 3' → '7-3'."""
    if not raw:
        return "—"
    cleaned = raw.replace(" ", "")
    return cleaned if cleaned else "—"


def _parse_standings() -> StandingsResponse:
    import statsapi

    raw = statsapi.standings_data()
    divisions: list[DivisionStandings] = []

    for div_data in raw.values():
        div_name = div_data.get("division_full_name", "")
        if not div_name:
            continue

        league = _DIVISION_LEAGUES.get(div_name, "")
        teams: list[TeamStanding] = []

        for rank_idx, team_data in enumerate(div_data.get("teams", []), start=1):
            team_name = team_data.get("name", "")
            abbrev = _ABBREVIATIONS.get(team_name, team_name[:3].upper())

            wins = int(team_data.get("w", 0) or 0)
            losses = int(team_data.get("l", 0) or 0)

            pct_raw = team_data.get("pct", "0")
            try:
                pct = float(pct_raw)
            except (ValueError, TypeError):
                pct = 0.0

            gb_raw = str(team_data.get("gb", "-") or "-")
            gb = "—" if gb_raw in ("-", "", "0.0", "0", None) else gb_raw

            streak = _format_streak(str(team_data.get("streak", "") or ""))
            last_10 = _format_last10(str(team_data.get("lastTen", "") or ""))

            home_rec = team_data.get("homeRecord") or None
            away_rec = team_data.get("awayRecord") or None

            wc_gb_raw = team_data.get("wildCardGb") or team_data.get("wildcard_gb") or None
            wc_gb = str(wc_gb_raw) if wc_gb_raw and str(wc_gb_raw) not in ("", "-", "E") else None

            teams.append(TeamStanding(
                team_name=team_name,
                team_abbreviation=abbrev,
                wins=wins,
                losses=losses,
                pct=pct,
                gb=gb,
                streak=streak,
                last_10=last_10,
                home_record=home_rec,
                away_record=away_rec,
                division_rank=rank_idx,
                wildcard_gb=wc_gb,
            ))

        divisions.append(DivisionStandings(
            division_name=div_name,
            league=league,
            teams=teams,
        ))

    # Sort: AL first, then alphabetically by division name
    divisions.sort(key=lambda d: (0 if d.league == "AL" else 1, d.division_name))

    return StandingsResponse(
        divisions=divisions,
        as_of=str(date.today()),
    )


@router.get("/", response_model=StandingsResponse)
async def get_standings():
    """Get current MLB standings from MLB Stats API (cached 10 minutes)."""
    global _cache

    now = time.time()
    if _cache["data"] is not None and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["data"]

    try:
        standings = _parse_standings()
        _cache = {"data": standings, "ts": now}
        return standings
    except Exception as e:
        logger.error(f"Failed to fetch standings: {e}")
        if _cache["data"] is not None:
            logger.info("Returning stale standings cache after error")
            return _cache["data"]
        return StandingsResponse(divisions=[], as_of=str(date.today()))
