import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import PipelineOrchestrator

class TestPipelineOrchestrator:
    
    @pytest.fixture
    def mock_components(self):
        """Mock all pipeline components."""
        with patch('src.pipeline.MLBDataFetcher') as fetcher, \
             patch('src.pipeline.SeriesTracker') as tracker, \
             patch('src.pipeline.MLBStatsAnalyzer') as analyzer, \
             patch('src.pipeline.ScriptGenerator') as script_gen, \
             patch('src.pipeline.AudioGenerator') as audio_gen, \
             patch('src.pipeline.VideoAssembler') as video_asm:
            
            yield {
                'fetcher': fetcher,
                'tracker': tracker,
                'analyzer': analyzer,
                'script_gen': script_gen,
                'audio_gen': audio_gen,
                'video_asm': video_asm
            }
    
    def test_pipeline_initialization(self, mock_components):
        """Test that pipeline initializes all components."""
        orchestrator = PipelineOrchestrator()
        
        assert orchestrator.fetcher is not None
        assert orchestrator.series_tracker is not None
        assert orchestrator.analyzer is not None
    
    def test_run_for_date_success(self, mock_components):
        """Test successful pipeline execution."""
        orchestrator = PipelineOrchestrator()
        
        # Mock game data
        orchestrator.fetcher.get_schedule = MagicMock(return_value=[
            {'game_id': 123, 'away_name': 'Yankees', 'home_name': 'Red Sox'}
        ])
        orchestrator.fetcher.get_game_data = MagicMock(return_value={'game_id': 123})
        
        # Mock other components
        orchestrator.series_tracker.get_video_type = MagicMock(return_value='series_middle')
        orchestrator.analyzer.analyze_game = MagicMock(return_value={'insights': []})
        orchestrator.script_generator.generate_script = MagicMock(return_value="Script")
        orchestrator.audio_generator.generate_audio = MagicMock(return_value="audio.mp3")
        orchestrator.chart_generator.generate_trend_chart = MagicMock(return_value="chart.png")
        orchestrator.video_assembler.assemble_video = MagicMock(return_value="video.mp4")
        
        result = orchestrator.run_for_date("2024-07-04", "Yankees")
        
        assert result == "video.mp4"
    
    def test_run_for_date_no_games(self, mock_components):
        """Test pipeline when no games are found."""
        orchestrator = PipelineOrchestrator()
        orchestrator.fetcher.get_schedule = MagicMock(return_value=[])
        
        result = orchestrator.run_for_date("2024-07-04", "Yankees")
        
        assert result is None
