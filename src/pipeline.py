import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import time

from src.data import MLBDataFetcher, SeriesTracker, PredictionDataProcessor
from src.analysis import MLBStatsAnalyzer
from src.models import PlayerPerformanceLSTM, PredictionExplainer
from src.content import ScriptGenerator, AudioGenerator
from src.video import AssetManager, ChartGenerator, VideoAssembler
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
                script_data={"prediction": prediction},
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
    
    def _find_team_game(self, games: list, team_name: str) -> Optional[Dict]:
        """Find a game for the specified team."""
        for game in games:
            if team_name.lower() in game.get('away_name', '').lower() or \
               team_name.lower() in game.get('home_name', '').lower():
                return game
        return None
