import logging
import time
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional

from src.data import MLBDataFetcher
from src.utils.exceptions import PipelineError

logger = logging.getLogger(__name__)


class WatcherError(PipelineError):
    """Raised when the game watcher encounters an error."""

    error_code = "WATCHER_ERROR"

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class GameWatcher:
    """
    Polls the MLB Stats API and detects game status transitions.
    When a game changes from 'In Progress' to 'Final', triggers a callback
    to start the video generation pipeline.
    """

    def __init__(
        self,
        poll_interval: int = 120,
        teams: Optional[List[str]] = None,
        on_game_final: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Args:
            poll_interval: Seconds between polls (default 120 = 2 minutes).
            teams: List of team names to watch. None watches all teams.
            on_game_final: Callback invoked when a game transitions to 'Final'.
        """
        self.poll_interval = poll_interval
        self.teams = teams
        self.on_game_final = on_game_final

        self._game_states: Dict[int, str] = {}
        self._running = False
        self._stop_event = threading.Event()
        self.fetcher = MLBDataFetcher(enable_cache=False)

    def start(self) -> None:
        """
        Main polling loop. Runs indefinitely until stop() is called.
        Blocks the calling thread.
        """
        self._running = True
        self._stop_event.clear()
        logger.info(
            f"GameWatcher started. Polling every {self.poll_interval}s. "
            f"Watching teams: {self.teams or 'ALL'}"
        )

        while self._running:
            try:
                final_games = self._poll_once()
                for game in final_games:
                    self._handle_game_final(game)
            except Exception as e:
                logger.error(f"Error during poll: {e}")

            # Wait for poll_interval or until stop is called
            if self._stop_event.wait(timeout=self.poll_interval):
                break

        logger.info("GameWatcher stopped.")

    def stop(self) -> None:
        """Signal the watcher to stop after the current poll cycle."""
        self._running = False
        self._stop_event.set()

    def _poll_once(self) -> List[Dict]:
        """
        Fetch today's schedule and detect games that transitioned to 'Final'.

        Returns:
            List of game dicts that just became Final.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        games = self.fetcher.get_schedule(start_date=today, end_date=today)

        if not games:
            return []

        # Filter to watched teams if specified
        if self.teams:
            games = self._filter_teams(games)

        newly_final = []

        for game in games:
            game_id = game.get("game_id")
            current_status = game.get("status", "")
            previous_status = self._game_states.get(game_id)

            # Update tracked state
            self._game_states[game_id] = current_status

            # Detect transition to Final
            if current_status == "Final" and previous_status != "Final":
                if previous_status is not None:
                    logger.info(
                        f"Game {game_id} transitioned: "
                        f"'{previous_status}' -> 'Final' "
                        f"({game.get('away_name', '?')} vs {game.get('home_name', '?')})"
                    )
                else:
                    logger.info(
                        f"Game {game_id} detected as Final on first poll "
                        f"({game.get('away_name', '?')} vs {game.get('home_name', '?')})"
                    )
                newly_final.append(game)

        if not newly_final:
            active = sum(
                1 for s in self._game_states.values()
                if s and s != "Final" and s != "Pre-Game"
            )
            logger.debug(
                f"Poll complete. {len(self._game_states)} games tracked, "
                f"{active} active, {len(newly_final)} newly final."
            )

        return newly_final

    def _handle_game_final(self, game: Dict) -> None:
        """
        Invoke the on_game_final callback for a completed game.
        Wraps the call in try/except to prevent one failure from stopping the watcher.
        """
        if not self.on_game_final:
            logger.warning("No on_game_final callback configured")
            return

        game_id = game.get("game_id", "unknown")
        away = game.get("away_name", "?")
        home = game.get("home_name", "?")

        try:
            logger.info(f"Triggering pipeline for game {game_id}: {away} vs {home}")
            self.on_game_final(game)
        except Exception as e:
            logger.error(
                f"Callback failed for game {game_id} ({away} vs {home}): {e}"
            )

    def _filter_teams(self, games: List[Dict]) -> List[Dict]:
        """Filter games to only include watched teams."""
        if not self.teams:
            return games

        watched = [t.lower() for t in self.teams]
        return [
            g for g in games
            if g.get("away_name", "").lower() in watched
            or g.get("home_name", "").lower() in watched
        ]
