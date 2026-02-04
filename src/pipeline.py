import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import time

from src.data import MLBDataFetcher, SeriesTracker, PredictionDataProcessor
from src.analysis import MLBStatsAnalyzer
from src.models import PlayerPerformanceLSTM, PredictionExplainer, TeamWinPredictor
from src.content import ScriptGenerator, AudioGenerator, ImageGenerator, ViralScriptGenerator
from src.video import AssetManager, ChartGenerator, VideoAssembler, CinematicEngine, ViralVideoEngine
from src.utils import CostTracker, MetricsCollector, AlertManager

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Orchestrates the entire video generation pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize all components
        logger.info("Initializing pipeline components...")
        
        self.fetcher = MLBDataFetcher()
        self.series_tracker = SeriesTracker(self.fetcher)
        self.analyzer = MLBStatsAnalyzer(self.fetcher)
        
        self.data_processor = PredictionDataProcessor(self.fetcher)
        self.model = PlayerPerformanceLSTM()
        self.explainer = PredictionExplainer()
        
        self.cost_tracker = CostTracker()
        self.script_generator = ScriptGenerator()
        self.audio_generator = AudioGenerator(cost_tracker=self.cost_tracker)
        
        self.asset_manager = AssetManager()
        self.chart_generator = ChartGenerator()
        self.video_assembler = VideoAssembler(self.asset_manager)

        # Cinematic pipeline components
        self.image_generator = ImageGenerator()
        self.cinematic_engine = CinematicEngine()

        # Monitoring components
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()

        logger.info("Pipeline initialized successfully.")
    
    def run_for_date(self, date: str, team_name: str = "Yankees") -> Optional[str]:
        """
        Run the complete pipeline for a specific date and team.
        
        Args:
            date: Date in YYYY-MM-DD format
            team_name: Team to generate video for
            
        Returns:
            Path to generated video, or None if failed
        """
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            logger.info(f"Starting pipeline for {team_name} on {date}")
            
            # Step 1: Fetch Game Data
            logger.info("Step 1: Fetching game data...")
            games = self.fetcher.get_schedule(start_date=date, end_date=date)
            
            if not games:
                logger.warning(f"No games found for {date}")
                error_msg = "No games found"
                return None
            
            # Find the team's game
            game = self._find_team_game(games, team_name)
            if not game:
                logger.warning(f"No game found for {team_name} on {date}")
                error_msg = f"No game found for {team_name}"
                return None
            
            game_data = self.fetcher.get_game_data(game['game_id'])

            # Merge schedule-level fields into game_data so downstream
            # components (script generator, video assembler) have real
            # team names, IDs, scores, and series status.
            game_data.update({
                "away_team": game.get("away_name", "Away Team"),
                "home_team": game.get("home_name", "Home Team"),
                "away_team_id": game.get("away_id"),
                "home_team_id": game.get("home_id"),
                "away_score": game.get("away_score", 0),
                "home_score": game.get("home_score", 0),
                "date": game.get("game_date", date),
                "series_status": game.get("series_status", ""),
            })

            # Step 2: Determine Video Type
            logger.info("Step 2: Determining video type...")
            video_type = self.series_tracker.get_video_type(game_data)
            
            # Step 3: Analyze Game
            logger.info("Step 3: Analyzing game...")
            analysis = self.analyzer.analyze_game(game_data)
            
            # Step 4: Generate Prediction
            logger.info("Step 4: Generating prediction...")
            # For MVP, we'll use dummy prediction data
            prediction = {
                "prediction": "Above Average",
                "confidence": "High",
                "reasons": ["Strong recent form", "Favorable matchup"]
            }
            
            # Step 5: Generate Script
            logger.info("Step 5: Generating script...")
            script = self.script_generator.generate_script(
                game_data=game_data,
                analysis=analysis,
                prediction=prediction,
                video_type=video_type
            )

            # Steps 6-7: Run audio, charts, and asset prefetch in parallel
            logger.info("Steps 6-7: Generating audio, charts, and prefetching assets in parallel...")
            audio_file = None
            charts = []

            def _generate_audio():
                audio_path = f"outputs/audio/{date}_{team_name}.mp3"
                return self.audio_generator.generate_audio(script, audio_path)

            def _generate_charts():
                result = []
                trend_path = self.chart_generator.generate_trend_chart(
                    data=[0.250, 0.260, 0.280, 0.290, 0.310],
                    labels=["G1", "G2", "G3", "G4", "G5"],
                    title="Last 5 Games Batting Average",
                    filename=f"{date}_trend.png"
                )
                if trend_path:
                    result.append(trend_path)
                return result

            def _prefetch_assets():
                """Pre-download team logos so video assembly doesn't block on network."""
                away_id = game_data.get("away_team_id", 147)
                home_id = game_data.get("home_team_id", 111)
                self.asset_manager.fetch_team_logo(away_id)
                self.asset_manager.fetch_team_logo(home_id)

            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_audio = pool.submit(_generate_audio)
                fut_charts = pool.submit(_generate_charts)
                fut_assets = pool.submit(_prefetch_assets)

                # Wait for all to complete
                audio_file = fut_audio.result()
                charts = fut_charts.result()
                fut_assets.result()  # just ensure it finished

            if not audio_file:
                logger.error("Audio generation failed")
                error_msg = "Audio generation failed"
                return None

            # Step 8: Assemble Video
            logger.info("Step 8: Assembling video...")
            video_path = self.video_assembler.assemble_video(
                script_data={
                    "prediction": prediction,
                    "away_team_id": game_data.get("away_team_id", 147),
                    "home_team_id": game_data.get("home_team_id", 111),
                },
                audio_path=audio_file,
                charts=charts,
                output_filename=f"{date}_{team_name}.mp4"
            )
            
            if video_path:
                logger.info(f"✅ Pipeline completed successfully! Video: {video_path}")
                success = True
                
                # Get final costs
                total_cost = self.cost_tracker.get_total_cost()
                logger.info(f"Total API cost: ${total_cost:.4f}")
                
                # Check cost threshold
                self.alerts.check_cost_threshold(total_cost, threshold=1.00)
                
                return video_path
            else:
                logger.error("Video assembly failed")
                error_msg = "Video assembly failed"
                return None
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            error_msg = str(e)
            import traceback
            traceback.print_exc()
            
            # Send error alert
            self.alerts.send_error_alert(
                error_message=str(e),
                context={"date": date, "team": team_name}
            )
            
            return None
        
        finally:
            # Log metrics regardless of success/failure
            duration = time.time() - start_time
            costs = {
                "gemini": 0,  # Free with Google One AI Premium
                "audio": 0,  # Free with local Qwen3-TTS
                "total": 0  # All free!
            }
            
            self.metrics.log_pipeline_run(
                date=date,
                team=team_name,
                success=success,
                duration=duration,
                costs=costs,
                error=error_msg
            )
    
    def run_cinematic_for_date(self, date: str, team_name: str = "Yankees") -> Optional[str]:
        """
        Run the cinematic pipeline for a specific date and team.

        Uses AI-generated images with Ken Burns motion effects instead of
        chart overlays and static backgrounds.

        Args:
            date: Date in YYYY-MM-DD format
            team_name: Team to generate video for

        Returns:
            Path to generated video, or None if failed.
            Also stores the cinematic script in self._last_cinematic_script
            so main.py can access video_metadata for YouTube upload.
        """
        start_time = time.time()
        success = False
        error_msg = None
        self._last_cinematic_script = None

        try:
            logger.info(f"Starting cinematic pipeline for {team_name} on {date}")

            # Step 1: Fetch Game Data
            logger.info("Step 1: Fetching game data...")
            games = self.fetcher.get_schedule(start_date=date, end_date=date)

            if not games:
                logger.warning(f"No games found for {date}")
                error_msg = "No games found"
                return None

            game = self._find_team_game(games, team_name)
            if not game:
                logger.warning(f"No game found for {team_name} on {date}")
                error_msg = f"No game found for {team_name}"
                return None

            game_data = self.fetcher.get_game_data(game['game_id'])

            game_data.update({
                "away_team": game.get("away_name", "Away Team"),
                "home_team": game.get("home_name", "Home Team"),
                "away_team_id": game.get("away_id"),
                "home_team_id": game.get("home_id"),
                "away_score": game.get("away_score", 0),
                "home_score": game.get("home_score", 0),
                "date": game.get("game_date", date),
                "series_status": game.get("series_status", ""),
            })

            # Step 2: Determine Video Type
            logger.info("Step 2: Determining video type...")
            video_type = self.series_tracker.get_video_type(game_data)

            # Step 3: Analyze Game
            logger.info("Step 3: Analyzing game...")
            analysis = self.analyzer.analyze_game(game_data)

            # Step 4: Generate Prediction
            logger.info("Step 4: Generating prediction...")
            prediction = {
                "prediction": "Above Average",
                "confidence": "High",
                "reasons": ["Strong recent form", "Favorable matchup"]
            }

            # Step 5: Generate Cinematic Script (JSON)
            logger.info("Step 5: Generating cinematic script...")
            script = self.script_generator.generate_cinematic_script(
                game_data=game_data,
                analysis=analysis,
                prediction=prediction,
                video_type=video_type,
            )
            self._last_cinematic_script = script

            scenes = script.get("scenes", [])
            if not scenes:
                logger.error("Cinematic script has no scenes")
                error_msg = "No scenes in cinematic script"
                return None

            # Step 6: Parallel generation of audio + images
            logger.info("Step 6: Generating scene audio and images in parallel...")

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_audio = pool.submit(
                    self._generate_scene_audio, scenes, date, team_name
                )
                fut_images = pool.submit(
                    self.image_generator.generate_scene_images, scenes
                )

                scene_audio = fut_audio.result()
                scene_images = fut_images.result()

            if not scene_audio:
                logger.error("Scene audio generation failed")
                error_msg = "Scene audio generation failed"
                return None

            # Step 7: Render cinematic video
            logger.info("Step 7: Rendering cinematic video...")
            video_path = self.cinematic_engine.render_video(
                script=script,
                scene_images=scene_images,
                scene_audio_paths=scene_audio,
                output_filename=f"{date}_{team_name}_cinematic.mp4",
            )

            if video_path:
                logger.info(f"Cinematic pipeline completed! Video: {video_path}")
                success = True

                total_cost = self.cost_tracker.get_total_cost()
                logger.info(f"Total API cost: ${total_cost:.4f}")
                self.alerts.check_cost_threshold(total_cost, threshold=1.00)

                return video_path
            else:
                logger.error("Cinematic video rendering failed")
                error_msg = "Cinematic video rendering failed"
                return None

        except Exception as e:
            logger.error(f"Cinematic pipeline failed: {e}")
            error_msg = str(e)
            import traceback
            traceback.print_exc()

            self.alerts.send_error_alert(
                error_message=str(e),
                context={"date": date, "team": team_name, "mode": "cinematic"}
            )

            return None

        finally:
            duration = time.time() - start_time
            costs = {
                "gemini": 0,
                "nano_banana": 0,
                "audio": 0,
                "total": 0,
            }

            self.metrics.log_pipeline_run(
                date=date,
                team=team_name,
                success=success,
                duration=duration,
                costs=costs,
                error=error_msg,
            )

    def _generate_scene_audio(
        self,
        scenes: list,
        date: str,
        team_name: str,
    ) -> Dict[int, str]:
        """
        Generate TTS audio for each scene's narration text.

        Returns:
            {scene_id: audio_path} mapping.
        """
        results = {}
        for scene in scenes:
            scene_id = scene.get("scene_id", 0)
            narration = scene.get("narration", "")
            if not narration:
                logger.warning(f"Scene {scene_id} has no narration, skipping audio")
                continue

            audio_path = f"outputs/audio/{date}_{team_name}_scene_{scene_id}.wav"
            try:
                result = self.audio_generator.generate_audio(narration, audio_path)
                if result:
                    results[scene_id] = result
                else:
                    logger.warning(f"Audio generation returned None for scene {scene_id}")
            except Exception as e:
                logger.error(f"Audio generation failed for scene {scene_id}: {e}")

        return results

    def _find_team_game(self, games: list, team_name: str) -> Optional[Dict]:
        """Find a game for the specified team."""
        for game in games:
            if team_name.lower() in game.get('away_name', '').lower() or \
               team_name.lower() in game.get('home_name', '').lower():
                return game
        return None

    def run_viral_for_date(self, date: str, team_name: str = "Yankees") -> Optional[str]:
        """
        Run the viral pipeline: fast-paced recap + win probability prediction.
        """
        start_time = time.time()
        success = False
        error_msg = None
        costs = {"gemini": 0.0, "tts": 0.0} # Placeholder until CostTracker is fully wired
        
        # Lazy load viral components
        self.viral_script_generator = ViralScriptGenerator()
        self.team_predictor = TeamWinPredictor()
        self.viral_engine = ViralVideoEngine()
        
        # New components
        from src.utils.video_validator import VideoValidator
        from src.utils.pipeline_monitor import PipelineMonitor
        validator = VideoValidator()
        monitor = PipelineMonitor()
        
        try:
            logger.info(f"Starting VIRAL pipeline for {team_name} on {date}")
            
            # 1. Fetch Game (Complete Data)
            logger.info("Step 1: Fetching COMPLETE game data...")
            games = self.fetcher.get_schedule(start_date=date, end_date=date)
            if not games: 
                logger.warning(f"No games found for {date}")
                return None
            
            game_summary = self._find_team_game(games, team_name)
            if not game_summary: 
                logger.warning(f"No game found for {team_name}")
                return None
            
            # Use new robust fetcher method
            game_data = self.fetcher.get_complete_game_data(game_summary['game_id'])
            
            # 2. Analyze (Analysis now effectively done during fetch/enrichment, but keeping analyzer for future)
            # analysis = self.analyzer.analyze_game(game_data)
            analysis = {}
            
            # 3. Predict Next Game
            logger.info("Step 3: Finding next game and generating prediction...")
            next_game = self.fetcher.get_next_game(team_name, date)
            prediction = None
            
            if next_game:
                logger.info(f"Next game found: {next_game['opponent']} on {next_game['date']}")
                
                # Use Mock Season Stats for MVP since we don't have historical DB yet
                prediction = self.team_predictor.predict(
                    team_stats={
                        "win_pct": 0.550, 
                        "last_10_win_pct": 0.600,
                        "run_diff_per_game": 0.5
                    }, 
                    opp_stats={
                        "win_pct": 0.480,
                        "last_10_win_pct": 0.400,
                        "run_diff_per_game": -0.2
                    },
                    is_home=next_game['is_home']
                )
            else:
                logger.warning("No next game found, using heuristic/demo prediction for video continuity.")
                prediction = self.team_predictor._fallback_prediction()
                prediction['note'] = "End of Season / Demo Mode"
            
            # 4. Generate Viral Script
            script = self.viral_script_generator.generate_viral_script(
                game_data=game_data,
                prediction=prediction,
                next_game_data=next_game
            )
            self._last_viral_script = script
            
            # 5. Parallel Audio/Assets
            scenes = script.get("scenes", [])
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_audio = pool.submit(self._generate_scene_audio, scenes, date, team_name)
                # Prefetch assets for top performers
                def _prefetch_images():
                    self.asset_manager.fetch_team_logo(game_data['home_team_id'])
                    self.asset_manager.fetch_team_logo(game_data['away_team_id'])
                    # Future: Fetch player heads here
                    
                fut_assets = pool.submit(_prefetch_images)
                
                scene_audio = fut_audio.result()
                fut_assets.result()

            # 6. Assemble Video
            video_path = self.viral_engine.render_video(
                script=script,
                game_data=game_data,
                prediction=prediction,
                scene_audio_paths=scene_audio,
                output_filename=f"{date}_{team_name}_viral.mp4"
            )
            
            if video_path:
                # 7. Validation
                try:
                    logger.info("Step 7: Validating video...")
                    validator.validate_video(video_path)
                    success = True
                    logger.info(f"Viral pipeline success: {video_path}")
                    return video_path
                except Exception as ve:
                    logger.error(f"Video validation failed: {ve}")
                    # We return the path anyway but marked as failed internally? 
                    # Or return None? Strict mode -> Return None.
                    error_msg = f"Validation failed: {ve}"
                    return None
            else:
                error_msg = "Video rendering failed"
                return None
                
        except Exception as e:
            logger.error(f"Viral pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            return None
        finally:
            # Metrics
            duration = time.time() - start_time
            monitor.log_run(
                date=date,
                team=team_name,
                mode="viral",
                success=success,
                duration_seconds=duration,
                costs=costs,
                error=error_msg
            )
