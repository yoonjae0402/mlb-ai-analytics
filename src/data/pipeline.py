"""MLB Data Pipeline — fetches real data from pybaseball + MLB Stats API.

Data sources (all free, no API key needed):
- pybaseball: Statcast data, FanGraphs stats, Baseball Reference
- MLB Stats API (via statsapi): live games, schedules, rosters, game logs
"""

import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import statsapi

from src.data.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# Rate limiting: minimum seconds between pybaseball calls
_PYBASEBALL_MIN_INTERVAL = 3.0
_last_pybaseball_call = 0.0

# MLB image URL templates
HEADSHOT_URL = "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/{mlb_id}/headshot/67/current"
TEAM_LOGO_URL = "https://www.mlbstatic.com/team-logos/{team_id}.svg"

# MiLB level mapping for statsapi
MILB_LEVELS = {
    "AAA": "Triple-A",
    "AA": "Double-A",
    "A+": "High-A",
    "A": "Single-A",
}


def _rate_limit():
    """Throttle pybaseball calls to avoid scraping bans."""
    global _last_pybaseball_call
    elapsed = time.time() - _last_pybaseball_call
    if elapsed < _PYBASEBALL_MIN_INTERVAL:
        time.sleep(_PYBASEBALL_MIN_INTERVAL - elapsed)
    _last_pybaseball_call = time.time()


def _retry(func, max_retries=3, backoff=2.0):
    """Retry with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = backoff ** attempt
            logger.warning(f"Retry {attempt+1}/{max_retries}: {e}. Waiting {wait:.1f}s")
            time.sleep(wait)


def get_headshot_url(mlb_id: int) -> str:
    """Generate MLB headshot URL for a player."""
    return HEADSHOT_URL.format(mlb_id=mlb_id)


def get_team_logo_url(team_id: int) -> str:
    """Generate team logo SVG URL."""
    return TEAM_LOGO_URL.format(team_id=team_id)


class MLBDataPipeline:
    """Fetches real MLB data and stores it in PostgreSQL."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache = CacheManager(cache_dir)

    def fetch_batting_stats(self, season: int) -> pd.DataFrame:
        """Fetch season batting stats from FanGraphs via pybaseball."""
        cache_key = f"batting_stats_{season}"
        cached = self.cache.get(cache_key, "season_stats")
        if cached is not None:
            return cached

        from pybaseball import batting_stats
        _rate_limit()
        logger.info(f"Fetching batting stats for {season} from FanGraphs...")
        df = _retry(lambda: batting_stats(season, qual=50))

        if df is not None and not df.empty:
            self.cache.set(cache_key, df)
        return df

    def fetch_pitching_stats(self, season: int) -> pd.DataFrame:
        """Fetch season pitching stats from FanGraphs via pybaseball."""
        cache_key = f"pitching_stats_{season}"
        cached = self.cache.get(cache_key, "season_stats")
        if cached is not None:
            return cached

        from pybaseball import pitching_stats
        _rate_limit()
        logger.info(f"Fetching pitching stats for {season} from FanGraphs...")
        df = _retry(lambda: pitching_stats(season, qual=20))

        if df is not None and not df.empty:
            self.cache.set(cache_key, df)
        return df

    def fetch_player_game_logs(self, player_id: int, season: int) -> list[dict]:
        """Fetch game-by-game hitting logs from MLB Stats API."""
        cache_key = f"game_logs_{player_id}_{season}"
        cached = self.cache.get(cache_key, "game_logs")
        if cached is not None:
            return cached.to_dict("records")

        logger.info(f"Fetching game logs for player {player_id}, season {season}")
        params = {
            "personIds": player_id,
            "hydrate": f"stats(group=[hitting],type=[gameLog],season={season})",
        }
        data = _retry(lambda: statsapi.get("people", params))

        logs = []
        if "people" in data and len(data["people"]) > 0:
            person_data = data["people"][0]
            if "stats" in person_data:
                for stat_group in person_data["stats"]:
                    if "splits" in stat_group:
                        logs.extend(stat_group["splits"])

        if logs:
            df = pd.DataFrame(logs)
            self.cache.set(cache_key, df)
        return logs

    def fetch_statcast_batter(
        self, player_id: int, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch pitch-by-pitch Statcast data for a batter."""
        cache_key = f"statcast_{player_id}_{start}_{end}"
        cached = self.cache.get(cache_key, "statcast")
        if cached is not None:
            return cached

        from pybaseball import statcast_batter
        _rate_limit()
        logger.info(f"Fetching Statcast data for batter {player_id} ({start} to {end})")
        df = _retry(lambda: statcast_batter(start, end, player_id))

        if df is not None and not df.empty:
            self.cache.set(cache_key, df)
        return df

    def fetch_schedule(self, game_date: str = None) -> list[dict]:
        """Fetch today's games and probable pitchers."""
        if game_date is None:
            game_date = date.today().strftime("%Y-%m-%d")
        try:
            return statsapi.schedule(start_date=game_date, end_date=game_date)
        except Exception as e:
            logger.error(f"Error fetching schedule: {e}")
            return []

    def fetch_schedule_range(self, start_date: str, end_date: str) -> list[dict]:
        """Fetch games for a date range (for calendar views)."""
        try:
            return statsapi.schedule(start_date=start_date, end_date=end_date)
        except Exception as e:
            logger.error(f"Error fetching schedule range: {e}")
            return []

    def fetch_all_players(self, season: int) -> list[dict]:
        """Fetch all players on 40-man rosters + top prospects.

        Returns a list of dicts with player info including MLB ID, name, team,
        position, and current level (MLB/AAA/AA/etc.).
        """
        cache_key = f"all_players_{season}"
        cached = self.cache.get(cache_key, "rosters")
        if cached is not None:
            return cached.to_dict("records")

        logger.info(f"Fetching all MLB rosters for {season}...")
        all_players = []

        try:
            teams = statsapi.get("teams", {"sportId": 1, "season": season})
            for team_info in teams.get("teams", []):
                team_id = team_info["id"]
                team_name = team_info.get("abbreviation", team_info.get("name", ""))

                try:
                    roster = statsapi.roster(team_id, rosterType="40Man", season=season)
                    # Parse roster text — statsapi.roster returns formatted text
                    # Use the API directly for structured data
                    roster_data = statsapi.get(
                        "team_roster",
                        {"teamId": team_id, "rosterType": "40Man", "season": season},
                    )
                    for entry in roster_data.get("roster", []):
                        person = entry.get("person", {})
                        pos = entry.get("position", {})
                        status = entry.get("status", {})
                        all_players.append({
                            "mlb_id": person.get("id"),
                            "name": person.get("fullName", ""),
                            "team": team_name,
                            "team_id": team_id,
                            "position": pos.get("abbreviation", ""),
                            "current_level": "MLB" if "Active" in status.get("description", "") else "MiLB",
                        })
                except Exception as e:
                    logger.warning(f"Error fetching roster for team {team_id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching teams: {e}")

        if all_players:
            df = pd.DataFrame(all_players)
            self.cache.set(cache_key, df)
        return all_players

    def fetch_milb_game_logs(self, player_id: int, season: int) -> list[dict]:
        """Fetch minor league game logs for a player."""
        cache_key = f"milb_game_logs_{player_id}_{season}"
        cached = self.cache.get(cache_key, "game_logs")
        if cached is not None:
            return cached.to_dict("records")

        logger.info(f"Fetching MiLB game logs for player {player_id}, season {season}")
        logs = []
        try:
            # Try each minor league level
            for level_code in ["11", "12", "13", "14"]:  # AAA, AA, High-A, A
                params = {
                    "personIds": player_id,
                    "hydrate": f"stats(group=[hitting],type=[gameLog],season={season},sportId={level_code})",
                }
                try:
                    data = statsapi.get("people", params)
                    if "people" in data and len(data["people"]) > 0:
                        person_data = data["people"][0]
                        if "stats" in person_data:
                            for stat_group in person_data["stats"]:
                                if "splits" in stat_group:
                                    for split in stat_group["splits"]:
                                        split["_level"] = level_code
                                    logs.extend(stat_group["splits"])
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Error fetching MiLB logs for {player_id}: {e}")

        if logs:
            df = pd.DataFrame(logs)
            self.cache.set(cache_key, df)
        return logs

    def seed_database(self, seasons: list[int] = None):
        """Fetch real MLB data and populate PostgreSQL tables.

        This imports from backend.db at call time to avoid circular imports
        when used as a standalone data pipeline.
        """
        if seasons is None:
            seasons = [2023, 2024]

        from backend.db.session import SyncSessionLocal, init_db_sync
        from backend.db.models import Player, PlayerStat, Team

        init_db_sync()
        session = SyncSessionLocal()

        try:
            # Seed teams first
            self._seed_teams(session)

            for season in seasons:
                logger.info(f"Seeding database with {season} data...")
                df = self.fetch_batting_stats(season)
                if df is None or df.empty:
                    logger.warning(f"No batting stats for {season}")
                    continue

                count = 0
                for _, row in df.iterrows():
                    # Resolve MLB ID from name
                    name = row.get("Name", "")
                    if not name:
                        continue

                    mlb_id = self._resolve_mlb_id(name)
                    if mlb_id is None:
                        continue

                    # Upsert player
                    player = session.query(Player).filter_by(mlb_id=mlb_id).first()
                    if player is None:
                        player = Player(
                            mlb_id=mlb_id,
                            name=name,
                            team=str(row.get("Team", "")),
                            position="",
                            bats=str(row.get("Bats", "")),
                            throws=str(row.get("Throws", "")),
                            headshot_url=get_headshot_url(mlb_id),
                            current_level="MLB",
                        )
                        session.add(player)
                        session.flush()
                    else:
                        # Update headshot URL if missing
                        if not player.headshot_url:
                            player.headshot_url = get_headshot_url(mlb_id)

                    # Fetch game logs for this player
                    logs = self.fetch_player_game_logs(mlb_id, season)
                    for log in logs:
                        stat_data = log.get("stat", {})
                        game_date_str = log.get("date", "")
                        if not game_date_str:
                            continue

                        try:
                            gd = datetime.strptime(game_date_str, "%Y-%m-%d").date()
                        except ValueError:
                            continue

                        existing = (
                            session.query(PlayerStat)
                            .filter_by(player_id=player.id, game_date=gd)
                            .first()
                        )
                        if existing:
                            continue

                        ps = PlayerStat(
                            player_id=player.id,
                            game_date=gd,
                            level="MLB",
                            batting_avg=_safe_float(stat_data.get("avg")),
                            obp=_safe_float(stat_data.get("obp")),
                            slg=_safe_float(stat_data.get("slg")),
                            woba=None,  # not in game logs; computed in feature builder
                            hits=_safe_int(stat_data.get("hits")),
                            home_runs=_safe_int(stat_data.get("homeRuns")),
                            rbi=_safe_int(stat_data.get("rbi")),
                            walks=_safe_int(stat_data.get("baseOnBalls")),
                            at_bats=_safe_int(stat_data.get("atBats")),
                            plate_appearances=_safe_int(stat_data.get("plateAppearances")),
                            doubles=_safe_int(stat_data.get("doubles")),
                            triples=_safe_int(stat_data.get("triples")),
                            strikeouts=_safe_int(stat_data.get("strikeOuts")),
                            stolen_bases=_safe_int(stat_data.get("stolenBases")),
                        )
                        session.add(ps)
                    count += 1
                    if count % 25 == 0:
                        session.commit()
                        logger.info(f"  Processed {count} players for {season}")

                session.commit()
                logger.info(f"Finished seeding {season}: {count} players")

        except Exception as e:
            session.rollback()
            logger.error(f"Error seeding database: {e}")
            raise
        finally:
            session.close()

    def _seed_teams(self, session):
        """Seed all 30 MLB teams with logos."""
        from backend.db.models import Team

        try:
            teams_data = statsapi.get("teams", {"sportId": 1})
            for t in teams_data.get("teams", []):
                team_id = t["id"]
                existing = session.query(Team).filter_by(mlb_id=team_id).first()
                if existing:
                    continue

                league_info = t.get("league", {})
                division_info = t.get("division", {})

                team = Team(
                    mlb_id=team_id,
                    name=t.get("name", ""),
                    abbreviation=t.get("abbreviation", ""),
                    league=league_info.get("abbreviation", ""),
                    division=division_info.get("name", "").replace(
                        "American League ", ""
                    ).replace("National League ", ""),
                    logo_url=get_team_logo_url(team_id),
                )
                session.add(team)
            session.commit()
            logger.info(f"Seeded {len(teams_data.get('teams', []))} teams")
        except Exception as e:
            logger.warning(f"Error seeding teams: {e}")
            session.rollback()

    def _resolve_mlb_id(self, name: str) -> Optional[int]:
        """Resolve a player name to MLB ID via statsapi lookup."""
        try:
            results = statsapi.lookup_player(name)
            if results:
                return results[0]["id"]
        except Exception:
            pass
        return None


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
