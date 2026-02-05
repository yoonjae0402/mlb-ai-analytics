import statsapi
# NOTE: pybaseball is imported but not actively used yet
# It's reserved for future Statcast metrics integration (Phase 1 enhancement)
import pybaseball
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from functools import wraps

from src.utils.exceptions import DataFetchError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def retry_on_api_error(max_retries: int = 3, backoff_factor: float = 2.0):
    """
    Decorator to retry API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff (seconds)
    """
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

            # Re-raise the last exception if all retries failed
            raise last_exception

        return wrapper
    return decorator

class MLBDataFetcher:
    """
    Handles fetching data from MLB APIs (statsapi, pybaseball).
    Includes caching to minimize API calls and handle rate limits.
    """
    

    def __init__(self, cache_dir: str = "./data/cache", enable_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # New: Photo Fetcher
        from src.data.player_photo_fetcher import PlayerPhotoFetcher
        from src.data.fallback_handler import FallbackHandler
        self.photo_fetcher = PlayerPhotoFetcher()
        self.fallback_handler = FallbackHandler
        
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

    @retry_on_api_error(max_retries=3, backoff_factor=2.0)
    def _fetch_schedule_from_api(self, start_date: str, end_date: str) -> List[Dict]:
        """Internal method to fetch schedule with retry logic."""
        logger.info(f"Fetching schedule from {start_date} to {end_date}")
        return statsapi.schedule(start_date=start_date, end_date=end_date)

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
        # Get boxscore content
        boxscore = statsapi.boxscore_data(game_id)
        # Get linescore
        linescore = statsapi.linescore(game_id)

        return {
            "game_id": game_id,
            "boxscore": boxscore,
            "linescore": linescore
        }

    def get_game_data(self, game_id: int) -> Dict:
        """
        Fetch detailed boxscore and linescore for a specific game.
        """
        cache_key = f"game_{game_id}"
        cached = self._load_from_cache(cache_key, "game_data")
        if cached:
            return cached

        try:
            combined_data = self._fetch_game_data_from_api(game_id)
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
            
    @retry_on_api_error(max_retries=3, backoff_factor=2.0)
    def _fetch_player_game_logs_from_api(self, player_id: int, season: int, group: str) -> List[Dict]:
        """Internal method to fetch player game logs with retry logic."""
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
        return logs

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
    def _fetch_standings_from_api(self, season: int) -> List[Dict]:
        """Internal method to fetch standings with retry logic."""
        # statsapi.standings returns a string by default, we want data if possible
        # statsapi.standings_data() returns structured data
        return statsapi.standings_data(season=season)

    def get_standings(self, season: int = None) -> List[Dict]:
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
            return []

    @retry_on_api_error(max_retries=3, backoff_factor=2.0)
    def get_next_game(self, team_name: str, after_date: str) -> Optional[Dict]:
        """
        Fetch the next scheduled game for a team after the given date.
        Searches up to 7 days ahead.
        """
        start_dt = datetime.strptime(after_date, "%Y-%m-%d")
        next_day_dt = start_dt + timedelta(days=1)
        end_dt = start_dt + timedelta(days=30) # Look ahead 30 days (extended for off-season/playoff gaps)
        
        start_str = next_day_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")
        
        try:
            schedule = self.get_schedule(start_str, end_str)
            for game in schedule:
                home = game.get("home_name", "")
                away = game.get("away_name", "")
                
                if team_name.lower() in home.lower() or team_name.lower() in away.lower():
                    # Found next game
                    # Enrich with opponent info
                    is_home = team_name.lower() in home.lower()
                    opponent = away if is_home else home
                    opponent_id = game.get("away_id") if is_home else game.get("home_id")
                    
                    return {
                        "game_id": game.get("game_id"),
                        "date": game.get("game_date"),
                        "opponent": opponent,
                        "opponent_id": opponent_id,
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

    def get_complete_game_data(self, game_id: int) -> Dict[str, Any]:
        """
        Fetch ALL necessary data for viral video generation.
        Uses the full game endpoint which provides gameData/liveData structure
        with team names, scores, boxscore, and play-by-play data.
        """
        try:
            # Use the game endpoint directly - it returns the full structure
            # with gameData (teams, venue) and liveData (linescore, boxscore, plays)
            full_data = statsapi.get("game", {"gamePk": game_id})

            if not full_data:
                 raise ValueError("Empty response from MLB API")
        except Exception as e:
            logger.error(f"Failed to fetch game {game_id}: {e}")
            raise DataFetchError(f"Failed to fetch game {game_id}: {e}")

        game_data = full_data.get('gameData', {})
        live_data = full_data.get('liveData', {})
        linescore = live_data.get('linescore', {})
        boxscore = live_data.get('boxscore', {})
        
        try:
            # 1. Parse Teams & Score
            teams_info = game_data.get('teams', {})
            home_meta = teams_info.get('home', {})
            away_meta = teams_info.get('away', {})
            
            home_team = home_meta.get('name', 'Home Team')
            away_team = away_meta.get('name', 'Away Team')
            
            home_score = linescore.get('teams', {}).get('home', {}).get('runs', 0)
            away_score = linescore.get('teams', {}).get('away', {}).get('runs', 0)

            # Determine winner
            winning_side = 'home' if home_score > away_score else 'away'
            winner = home_team if home_score > away_score else away_team
            loser = away_team if home_score > away_score else home_team

            # 2. Extract Top Performers (with team info, pitcher from winning team only)
            top_hitter = self._get_top_hitter(boxscore, full_data)
            top_pitcher = self._get_top_pitcher(boxscore, full_data, winning_side=winning_side)

            # 3. Key Moments
            key_moment = self._find_key_moment(live_data)
            if not key_moment:
                key_moment = {"description": f"{winner} secures the victory!", "score_change": "Final"}

            top_performers = [top_hitter, top_pitcher] # Keep list for backward compat if needed
            
            complete_data = {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_team_id": home_meta.get('id'),
                "away_team_id": away_meta.get('id'),
                "winner": winner,
                "loser": loser,
                "winning_side": winning_side,
                "top_hitter": top_hitter,
                "top_pitcher": top_pitcher,
                "key_moment": key_moment,
                "standings_impact": self._compute_standings_impact(winner, full_data),
                "boxscore": boxscore
            }
            
            # Legacy fields for validator compatibility if it checks 'top_performers'
            complete_data['top_performers'] = top_performers 
            
            return complete_data

        except Exception as e:
            logger.error(f"Failed to parse complete game data: {e}")
            raise DataFetchError(f"Data parsing failed: {e}")

    def _compute_standings_impact(self, winner: str, game_data: dict) -> str:
        """Generate dynamic standings impact text from actual standings."""
        try:
            game_date_str = game_data.get('gameData', {}).get('datetime', {}).get('dateTime', '')
            season = int(game_date_str[:4]) if game_date_str else datetime.now().year

            standings = self.get_standings(season)
            if not standings:
                return "Big win in the standings race"

            for div_id, div_data in standings.items():
                teams = div_data.get('teams', [])
                for team in teams:
                    if winner.lower() in team.get('name', '').lower():
                        wins = team.get('w', 0)
                        losses = team.get('l', 0)
                        div_rank = team.get('div_rank', '')
                        gb = team.get('gb', '-')
                        wc_gb = team.get('wc_gb', '')
                        div_name = div_data.get('div_name', '')

                        if div_rank == '1' or gb == '-':
                            return f"Holds first place in the {div_name.split()[-1]} ({wins}-{losses})"
                        elif gb != '-' and float(str(gb).replace('+', '')) <= 3.0:
                            return f"Closes to {gb} GB in the {div_name.split()[-1]} ({wins}-{losses})"
                        elif wc_gb and wc_gb != '-' and float(str(wc_gb).replace('+', '')) <= 5.0:
                            return f"Wild Card push - {wc_gb} GB ({wins}-{losses})"
                        else:
                            return f"Improves to {wins}-{losses} on the season"

            return "Big win in the standings race"
        except Exception as e:
            logger.warning(f"Could not compute standings impact: {e}")
            return "Big win in the standings race"

    def _get_top_hitter(self, boxscore: Dict, game_data: Dict = None) -> Dict:
        """Find player with most offensive impact (Hits + HR + RBI). Includes team info."""
        best_player = None
        best_score = -1

        teams = boxscore.get('teams', {})
        teams_info = game_data.get('gameData', {}).get('teams', {}) if game_data else {}

        for side in ['home', 'away']:
            team_name = teams_info.get(side, {}).get('name', side.title())
            players = teams.get(side, {}).get('players', {})

            for pid, p_data in players.items():
                stats = p_data.get('stats', {}).get('batting', {})
                if not stats: continue

                # Simple Impact Score
                hits = stats.get('hits', 0)
                hr = stats.get('homeRuns', 0)
                rbi = stats.get('rbi', 0)
                impact = (hits * 1) + (hr * 3) + (rbi * 2)

                if impact > best_score:
                    best_score = impact
                    stat_line = f"{hits} Hits, {hr} HR, {rbi} RBI"
                    best_player = {
                        "name": p_data.get('person', {}).get('fullName', 'Unknown'),
                        "team": team_name,
                        "team_side": side,
                        "stats": stat_line,
                        "impact": f"Huge {hr} HR performance" if hr > 0 else f"{hits} Hits led the way",
                        "id": p_data.get('person', {}).get('id')
                    }

        if not best_player:
            return self.fallback_handler.get_generic_player_data("Hitter")

        best_player['photo_url'] = self.photo_fetcher.fetch_player_photo(best_player['id'], best_player['name'])
        return best_player

    def _get_top_pitcher(self, boxscore: Dict, game_data: Dict = None, winning_side: str = None) -> Dict:
        """
        Find best pitcher. If winning_side is provided, only considers that team.
        Includes team info in the result.
        """
        best_pitcher = None
        best_score = -1

        teams = boxscore.get('teams', {})
        teams_info = game_data.get('gameData', {}).get('teams', {}) if game_data else {}

        # If winning_side specified, only look at that team
        sides_to_check = [winning_side] if winning_side else ['home', 'away']

        for side in sides_to_check:
            team_name = teams_info.get(side, {}).get('name', side.title())
            players = teams.get(side, {}).get('players', {})

            for pid, p_data in players.items():
                stats = p_data.get('stats', {}).get('pitching', {})
                if not stats: continue

                # Score based on K, IP, ER
                k = stats.get('strikeOuts', 0)
                ip = float(stats.get('inningsPitched', 0.0))
                er = stats.get('earnedRuns', 100)

                score = (k * 2) + (ip * 1) - (er * 2)
                if score > best_score:
                    best_score = score
                    stat_line = f"{ip} IP, {k} K, {er} ER"
                    best_pitcher = {
                        "name": p_data.get('person', {}).get('fullName', 'Unknown'),
                        "team": team_name,
                        "team_side": side,
                        "stats": stat_line,
                        "impact": "Dominant on the mound",
                        "id": p_data.get('person', {}).get('id')
                    }

        if not best_pitcher:
            return self.fallback_handler.get_generic_player_data("Pitcher")

        best_pitcher['photo_url'] = self.photo_fetcher.fetch_player_photo(best_pitcher['id'], best_pitcher['name'])
        return best_pitcher

    def _find_key_moment(self, live_data: Dict) -> Dict:
        """Find game changing play (highest wpa or scoring play)."""
        plays = live_data.get('plays', {}).get('allPlays', [])
        if not plays:
            return None
            
        # Simplistic: Find last scoring play or biggest hit
        best_play = None
        # Iterate backwards
        for play in reversed(plays):
            event = play.get('result', {}).get('event', '')
            if 'Home Run' in event or 'Score' in event:
                return {
                    "description": play.get('result', {}).get('description', 'Key Play'),
                    "score_change": "Game Changer", # Calculate actual score change if needed
                    "inning": play.get('about', {}).get('halfInning', '').title() + " " + str(play.get('about', {}).get('inning', ''))
                }
        
        # Default to last play
        if plays:
            last = plays[-1]
            return {
                "description": last.get('result', {}).get('description', 'Final Out'),
                "score_change": "Final Out",
                "inning": "9th"
            }
        
        return {"description": "Game End", "score_change": "Final", "inning": "9th"}

if __name__ == "__main__":
    # Simple test
    fetcher = MLBDataFetcher()
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching schedule for {today}...")
    games = fetcher.get_schedule(today)
    if games:
        print(f"Found {len(games)} games.")
        gid = games[0]['game_id']
        print(f"Fetching complete data for {gid}...")
        try:
            data = fetcher.get_complete_game_data(gid)
            print("Top Performers:")
            for p in data['top_performers']:
                print(f"- {p['name']}: {p['stats']}")
        except Exception as e:
            print(f"Error: {e}")