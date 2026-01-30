import statsapi
import pybaseball
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLBDataFetcher:
    """
    Handles fetching data from MLB APIs (statsapi, pybaseball).
    Includes caching to minimize API calls and handle rate limits.
    """
    
    def __init__(self, cache_dir: str = "./data/cache", enable_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache expiration times (in seconds)
        self.CACHE_EXPIRY = {
            "schedule": 3600,       # 1 hour
            "game_data": 300,       # 5 minutes (live games update frequently)
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

    def get_schedule(self, start_date: str, end_date: str = None) -> List[Dict]:
        """
        Fetch schedule for a date range.
        Dates in 'MM/DD/YYYY' format (statsapi default).
        """
        if end_date is None:
            end_date = start_date
            
        cache_key = f"schedule_{start_date}_{end_date}"
        cached = self._load_from_cache(cache_key, "schedule")
        if cached:
            return cached
            
        try:
            logger.info(f"Fetching schedule from {start_date} to {end_date}")
            schedule = statsapi.schedule(start_date=start_date, end_date=end_date)
            self._save_to_cache(cache_key, schedule)
            return schedule
        except Exception as e:
            logger.error(f"Error fetching schedule: {e}")
            return []

    def get_game_data(self, game_id: int) -> Dict:
        """
        Fetch detailed boxscore and linescore for a specific game.
        """
        cache_key = f"game_{game_id}"
        cached = self._load_from_cache(cache_key, "game_data")
        if cached:
            return cached
            
        try:
            logger.info(f"Fetching game data for {game_id}")
            # Get boxscore content
            boxscore = statsapi.boxscore_data(game_id)
            # Get linescore
            linescore = statsapi.linescore(game_id)
            
            combined_data = {
                "game_id": game_id,
                "boxscore": boxscore,
                "linescore": linescore
            }
            
            # Only cache if game is Final to prevent stale live data in long-term cache
            # For this simple versions, we cache everything but with short expiry (5 mins)
            self._save_to_cache(cache_key, combined_data)
            return combined_data
        except Exception as e:
            logger.error(f"Error fetching game data for {game_id}: {e}")
            return {}

    def get_player_stats_last_n_games(self, player_id: int, n: int = 10) -> pd.DataFrame:
        """
        Fetch player stats for the last N games.
        Uses pybaseball for statcast/game log data as it's often more granular for analysis.
        Note: Mixing IDs can be tricky. statsapi uses MLBAM ID. pybaseball usually supports it.
        """
        # This is a placeholder for the more complex logic needed to fetch last N games 
        # across seasons. For now, we'll try to use pybaseball to get game logs.
        
        # For simplicity in Phase 1, we might just look up season stats or recent game logs logic
        # strictly via statsapi if pybaseball is too heavy/slow for individual queries.
        # Let's stick to statsapi for individual player lookups for consistency first.
        
        cache_key = f"player_{player_id}_last_{n}"
        cached = self._load_from_cache(cache_key, "player_stats")
        if cached:
             return pd.DataFrame(cached)

        try:
            # Placeholder for now
            pass
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            
        return pd.DataFrame()
            
    def get_player_game_logs(self, player_id: int, season: int = None, group: str = "hitting") -> List[Dict]:
        """
        Fetch game logs for a player in a specific season.
        """
        if not season:
            season = datetime.now().year
            
        cache_key = f"player_{player_id}_log_{season}_{group}"
        cached = self._load_from_cache(cache_key, "player_stats")
        if cached:
            return cached
            
        try:
            logger.info(f"Fetching game logs for player {player_id}, season {season}")
            # Use statsapi.get to hit the hydrating endpoint
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
                        
            # Transform to a flatter, usable format if needed, or return raw splits
            # For now, let's just return the splits which contain 'stat' and 'date'
            
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


    def get_standings(self, season: int = None) -> List[Dict]:
        """Fetch standings for a season."""
        if not season:
            season = datetime.now().year
            
        cache_key = f"standings_{season}"
        cached = self._load_from_cache(cache_key, "standings")
        if cached:
            return cached
            
        try:
            # statsapi.standings returns a string by default, we want data if possible
            # statsapi.standings_data() returns structured data
            standings = statsapi.standings_data(season=season)
            self._save_to_cache(cache_key, standings)
            return standings
        except Exception as e:
            logger.error(f"Error fetching standings: {e}")
            return []

if __name__ == "__main__":
    # Simple test
    fetcher = MLBDataFetcher()
    today = datetime.now().strftime("%m/%d/%Y")
    print(f"Fetching schedule for {today}...")
    games = fetcher.get_schedule(today)
    print(f"Found {len(games)} games.")
    if games:
        print(f"First game: {games[0]['summary']}")