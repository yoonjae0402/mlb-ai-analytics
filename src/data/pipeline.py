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
            "personId": player_id,
            "stats": "gameLog",
            "group": "hitting",
            "season": season,
        }
        data = _retry(lambda: statsapi.get("people_stats", params))

        logs = []
        if "stats" in data:
            for stat_group in data["stats"]:
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

    def seed_database(self, seasons: list[int] = None):
        """Fetch real MLB data and populate PostgreSQL tables.

        This imports from backend.db at call time to avoid circular imports
        when used as a standalone data pipeline.
        """
        if seasons is None:
            seasons = [2023, 2024]

        from backend.db.session import SyncSessionLocal, init_db_sync
        from backend.db.models import Player, PlayerStat

        init_db_sync()
        session = SyncSessionLocal()

        try:
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
                        )
                        session.add(player)
                        session.flush()

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
