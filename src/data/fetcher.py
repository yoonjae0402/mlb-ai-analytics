"""
MLB Data Fetcher

Fetches data from MLB Stats API with caching support.
"""

import statsapi
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from functools import wraps


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Exception raised when data fetching fails."""
    pass


def retry_on_api_error(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator to retry API calls with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        wait_time = backoff_factor ** attempt
                        logger.warning(
                            f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"API call failed after {max_retries + 1} attempts: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


class MLBDataFetcher:
    """
    Handles fetching data from MLB Stats API.
    Includes caching to minimize API calls and handle rate limits.
    """

    def __init__(self, cache_dir: str = "./data/cache", enable_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache expiration times (in seconds)
        self.CACHE_EXPIRY = {
            "schedule": 3600,       # 1 hour
            "game_data": 300,       # 5 minutes
            "player_stats": 86400,  # 24 hours
            "standings": 3600       # 1 hour
        }

    def _get_cache_path(self, key: str) -> Path:
        """Generate a valid file path for a cache key."""
        safe_key = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _load_from_cache(self, key: str, cache_type: str) -> Optional[Any]:
        """Load data from cache if it exists and is not expired."""
        if not self.enable_cache:
            return None

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            timestamp = data.get('timestamp')
            if not timestamp:
                return None

            age = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
            expiry = self.CACHE_EXPIRY.get(cache_type, 3600)

            if age < expiry:
                logger.debug(f"Cache hit for {key}")
                return data.get('payload')
            else:
                logger.debug(f"Cache expired for {key}")
                return None
        except Exception as e:
            logger.warning(f"Failed to load cache for {key}: {e}")
            return None

    def _save_to_cache(self, key: str, payload: Any):
        """Save data to cache."""
        if not self.enable_cache:
            return

        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'payload': payload
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {key}: {e}")

    @retry_on_api_error(max_retries=3, backoff_factor=2.0)
    def _fetch_schedule_from_api(self, start_date: str, end_date: str) -> List[Dict]:
        """Internal method to fetch schedule with retry logic."""
        logger.info(f"Fetching schedule from {start_date} to {end_date}")
        return statsapi.schedule(start_date=start_date, end_date=end_date)

    def get_schedule(self, start_date: str, end_date: str = None) -> List[Dict]:
        """Fetch schedule for a date range."""
        if end_date is None:
            end_date = start_date

        cache_key = f"schedule_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_key, "schedule")
        if cached:
            return cached

        try:
            schedule = self._fetch_schedule_from_api(start_date, end_date)
            self._save_to_cache(cache_key, schedule)
            return schedule
        except Exception as e:
            logger.error(f"Error fetching schedule: {e}")
            return []

    @retry_on_api_error(max_retries=3, backoff_factor=2.0)
    def _fetch_game_data_from_api(self, game_id: int) -> Dict:
        """Internal method to fetch game data with retry logic."""
        logger.info(f"Fetching game data for {game_id}")
        boxscore = statsapi.boxscore_data(game_id)
        linescore = statsapi.linescore(game_id)

        return {
            "game_id": game_id,
            "boxscore": boxscore,
            "linescore": linescore
        }

    def get_game_data(self, game_id: int) -> Dict:
        """Fetch detailed boxscore and linescore for a specific game."""
        cache_key = f"game_{game_id}"
        cached = self._load_from_cache(cache_key, "game_data")
        if cached:
            return cached

        try:
            combined_data = self._fetch_game_data_from_api(game_id)
            self._save_to_cache(cache_key, combined_data)
            return combined_data
        except Exception as e:
            logger.error(f"Error fetching game data for {game_id}: {e}")
            return {}

    @retry_on_api_error(max_retries=3, backoff_factor=2.0)
    def _fetch_player_game_logs_from_api(self, player_id: int, season: int, group: str) -> List[Dict]:
        """Internal method to fetch player game logs with retry logic."""
        logger.info(f"Fetching game logs for player {player_id}, season {season}")
        params = {
            "personId": player_id,
            "stats": "gameLog",
            "group": group,
            "season": season
        }
        data = statsapi.get("people_stats", params)

        logs = []
        if "stats" in data:
            for stat_group in data["stats"]:
                if "splits" in stat_group:
                    logs.extend(stat_group["splits"])
        return logs

    def get_player_game_logs(self, player_id: int, season: int = None, group: str = "hitting") -> List[Dict]:
        """Fetch game logs for a player in a specific season."""
        if not season:
            season = datetime.now().year

        cache_key = f"player_{player_id}_log_{season}_{group}"
        cached = self._load_from_cache(cache_key, "player_stats")
        if cached:
            return cached

        try:
            logs = self._fetch_player_game_logs_from_api(player_id, season, group)
            self._save_to_cache(cache_key, logs)
            return logs
        except Exception as e:
            logger.error(f"Error fetching player game logs: {e}")
            return []

    def get_player_id(self, player_name: str) -> Optional[int]:
        """Lookup player ID by name."""
        try:
            players = statsapi.lookup_player(player_name)
            if players:
                return players[0]["id"]
            return None
        except Exception:
            return None

    @retry_on_api_error(max_retries=3, backoff_factor=2.0)
    def _fetch_standings_from_api(self, season: int) -> Dict:
        """Internal method to fetch standings with retry logic."""
        return statsapi.standings_data(season=season)

    def get_standings(self, season: int = None) -> Dict:
        """Fetch standings for a season."""
        if not season:
            season = datetime.now().year

        cache_key = f"standings_{season}"
        cached = self._load_from_cache(cache_key, "standings")
        if cached:
            return cached

        try:
            standings = self._fetch_standings_from_api(season)
            self._save_to_cache(cache_key, standings)
            return standings
        except Exception as e:
            logger.error(f"Error fetching standings: {e}")
            return {}

    @retry_on_api_error(max_retries=3, backoff_factor=2.0)
    def get_next_game(self, team_name: str, after_date: str) -> Optional[Dict]:
        """Fetch the next scheduled game for a team after the given date."""
        start_dt = datetime.strptime(after_date, "%Y-%m-%d")
        next_day_dt = start_dt + timedelta(days=1)
        end_dt = start_dt + timedelta(days=30)

        start_str = next_day_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        try:
            schedule = self.get_schedule(start_str, end_str)
            for game in schedule:
                home = game.get("home_name", "")
                away = game.get("away_name", "")

                if team_name.lower() in home.lower() or team_name.lower() in away.lower():
                    is_home = team_name.lower() in home.lower()
                    opponent = away if is_home else home

                    return {
                        "game_id": game.get("game_id"),
                        "date": game.get("game_date"),
                        "opponent": opponent,
                        "is_home": is_home,
                        "venue": game.get("venue_name"),
                        "status": game.get("status"),
                        "home_prob_pitcher": game.get("home_probable_pitcher", "TBD"),
                        "away_prob_pitcher": game.get("away_probable_pitcher", "TBD")
                    }

            logger.info(f"No upcoming game found for {team_name} in next 30 days.")
            return None

        except Exception as e:
            logger.error(f"Error fetching next game: {e}")
            return None


if __name__ == "__main__":
    # Simple test
    fetcher = MLBDataFetcher()
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching schedule for {today}...")
    games = fetcher.get_schedule(today)
    print(f"Found {len(games)} games.")
