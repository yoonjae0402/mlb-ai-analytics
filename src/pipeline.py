import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

from src.data import MLBDataFetcher
from src.content import AudioGenerator, ImageGenerator, ViralScriptGenerator
from src.video import AssetManager, ViralVideoEngine
from src.utils import CostTracker, MetricsCollector, AlertManager

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates game recap video generation.
    Produces game-neutral ESPN-style viral videos.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        logger.info("Initializing pipeline components...")

        self.fetcher = MLBDataFetcher()
        self.cost_tracker = CostTracker()
        self.audio_generator = AudioGenerator(cost_tracker=self.cost_tracker)
        self.asset_manager = AssetManager()
        self.image_generator = ImageGenerator()
        self.script_generator = ViralScriptGenerator()
        self.video_engine = ViralVideoEngine()

        self.metrics = MetricsCollector()
        self.alerts = AlertManager()

        # Store last script for upload metadata
        self._last_script = None

        logger.info("Pipeline initialized successfully.")

    def run_for_game(self, game_id: int) -> Optional[str]:
        """
        Generate a game-neutral recap video for a specific game.

        Args:
            game_id: MLB game ID

        Returns:
            Path to generated video, or None if failed
        """
        start_time = time.time()
        success = False
        error_msg = None

        # Lazy load validation
        from src.utils.video_validator import VideoValidator
        from src.utils.pipeline_monitor import PipelineMonitor
        validator = VideoValidator()
        monitor = PipelineMonitor()

        try:
            logger.info(f"Starting pipeline for game {game_id}")

            # 1. Fetch complete game data
            logger.info("Step 1: Fetching game data...")
            game_data = self.fetcher.get_complete_game_data(game_id)

            if not game_data:
                logger.error(f"Failed to fetch game data for {game_id}")
                return None

            home_team = game_data.get('home_team', 'Home')
            away_team = game_data.get('away_team', 'Away')
            logger.info(f"Game: {away_team} @ {home_team}")

            # 2. Generate script
            logger.info("Step 2: Generating script...")
            script = self.script_generator.generate_viral_script(game_data)
            self._last_script = script

            if not script or not script.get('scenes'):
                logger.error("Script generation failed")
                return None

            # 3. Extract emphasis words for TTS
            emphasis_words = self._extract_emphasis_words(game_data)
            logger.info(f"TTS emphasis words: {emphasis_words}")

            # 4. Generate audio and prefetch assets in parallel
            logger.info("Step 3: Generating audio and prefetching assets...")
            scenes = script.get("scenes", [])

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_audio = pool.submit(
                    self._generate_scene_audio, scenes, game_id
                )
                fut_assets = pool.submit(self._prefetch_assets, game_data)

                scene_audio = fut_audio.result()
                fut_assets.result()

            if not scene_audio:
                logger.error("Audio generation failed")
                return None

            # 5. Render video
            logger.info("Step 4: Rendering video...")
            output_filename = f"{game_id}_{away_team.split()[-1]}_vs_{home_team.split()[-1]}.mp4"

            video_path = self.video_engine.render_video(
                script=script,
                game_data=game_data,
                prediction={},  # No prediction for game-neutral videos
                scene_audio_paths=scene_audio,
                output_filename=output_filename
            )

            if video_path:
                # 6. Validate
                logger.info("Step 5: Validating video...")
                try:
                    validator.validate_video(video_path)
                    success = True
                    logger.info(f"Pipeline success: {video_path}")
                    return video_path
                except Exception as ve:
                    logger.error(f"Video validation failed: {ve}")
                    error_msg = f"Validation failed: {ve}"
                    return None
            else:
                error_msg = "Video rendering failed"
                return None

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            return None

        finally:
            duration = time.time() - start_time
            monitor.log_run(
                date=datetime.now().strftime("%Y-%m-%d"),
                team="game_neutral",
                mode="recap",
                success=success,
                duration_seconds=duration,
                costs={"total": 0},
                error=error_msg
            )

    def run_for_date(self, date: str, game_id: Optional[int] = None) -> List[str]:
        """
        Generate recap videos for games on a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            game_id: Optional specific game ID. If None, processes all final games.

        Returns:
            List of paths to generated videos
        """
        logger.info(f"Processing games for {date}")

        # Fetch schedule
        games = self.fetcher.get_schedule(start_date=date, end_date=date)

        if not games:
            logger.warning(f"No games found for {date}")
            return []

        # Filter to final games only
        final_games = [g for g in games if g.get('status') == 'Final']

        if not final_games:
            logger.warning(f"No completed games found for {date}")
            return []

        logger.info(f"Found {len(final_games)} completed games")

        # If specific game_id provided, filter to just that game
        if game_id:
            final_games = [g for g in final_games if g.get('game_id') == game_id]
            if not final_games:
                logger.warning(f"Game {game_id} not found or not final")
                return []

        # Process each game
        video_paths = []
        for game in final_games:
            gid = game.get('game_id')
            away = game.get('away_name', 'Away')
            home = game.get('home_name', 'Home')
            logger.info(f"Processing: {away} @ {home} (ID: {gid})")

            video_path = self.run_for_game(gid)
            if video_path:
                video_paths.append(video_path)

        return video_paths

    def _generate_scene_audio(self, scenes: list, game_id: int) -> Dict[int, str]:
        """Generate TTS audio for each scene."""
        results = {}
        for scene in scenes:
            scene_id = scene.get("scene_id", 0)
            narration = scene.get("narration", "")

            if not narration:
                logger.warning(f"Scene {scene_id} has no narration, skipping")
                continue

            audio_path = f"outputs/audio/{game_id}_scene_{scene_id}.wav"
            try:
                result = self.audio_generator.generate_audio(narration, audio_path)
                if result:
                    results[scene_id] = result
                else:
                    logger.warning(f"Audio generation returned None for scene {scene_id}")
            except Exception as e:
                logger.error(f"Audio generation failed for scene {scene_id}: {e}")

        return results

    def _prefetch_assets(self, game_data: Dict):
        """Pre-download team logos and player headshots."""
        try:
            self.asset_manager.fetch_team_logo(game_data.get('home_team_id'))
            self.asset_manager.fetch_team_logo(game_data.get('away_team_id'))

            # Prefetch player headshots
            for player_key in ['top_hitter', 'top_pitcher']:
                player = game_data.get(player_key, {})
                player_id = player.get('id')
                if player_id:
                    self.asset_manager.fetch_player_headshot(player_id)
        except Exception as e:
            logger.warning(f"Asset prefetch error: {e}")

    @staticmethod
    def _extract_emphasis_words(game_data: Dict) -> list:
        """Extract team and player names for TTS emphasis."""
        words = []

        for field in ['home_team', 'away_team', 'winner', 'loser']:
            name = game_data.get(field, '')
            if name:
                words.extend(name.split())

        for player_field in ['top_hitter', 'top_pitcher']:
            player = game_data.get(player_field, {})
            name = player.get('name', '')
            if name:
                words.extend(name.split())

        return list(set(w for w in words if len(w) > 2))
