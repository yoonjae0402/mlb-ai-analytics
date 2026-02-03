#!/usr/bin/env python3
"""
Verification script for Phase 5: Automation

This script tests the complete pipeline orchestration without
actually making API calls or uploading to YouTube.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PipelineOrchestrator

def verify_phase5():
    print("🚀 Verifying MLB Video Pipeline - Phase 5: Automation 🚀")
    print()
    
    # Mock all external dependencies
    with patch('src.pipeline.MLBDataFetcher') as mock_fetcher_class, \
         patch('src.pipeline.ScriptGenerator') as mock_script_class, \
         patch('src.pipeline.AudioGenerator') as mock_audio_class, \
         patch('src.pipeline.VideoAssembler') as mock_video_class:
        
        print("1. Initializing Pipeline Orchestrator...")
        orchestrator = PipelineOrchestrator()
        print("   ✓ Orchestrator initialized")
        
        print("\n2. Mocking Component Responses...")
        
        # Mock fetcher responses
        mock_fetcher = orchestrator.fetcher
        mock_fetcher.get_schedule = MagicMock(return_value=[
            {
                'game_id': 12345,
                'away_name': 'New York Yankees',
                'home_name': 'Boston Red Sox',
                'away_score': 5,
                'home_score': 3
            }
        ])
        mock_fetcher.get_game_data = MagicMock(return_value={
            'game_id': 12345,
            'away_team': 'Yankees',
            'home_team': 'Red Sox',
            'away_score': 5,
            'home_score': 3
        })
        
        # Mock other components
        orchestrator.series_tracker.get_video_type = MagicMock(return_value='series_middle')
        orchestrator.analyzer.analyze_game = MagicMock(return_value={
            'insights': ['Dominant victory'],
            'top_performers': [{'player': 'Judge', 'highlight': '2 HR'}]
        })
        orchestrator.script_generator.generate_script = MagicMock(
            return_value="Yankees win 5-3! Judge crushes 2 home runs!"
        )
        orchestrator.audio_generator.generate_audio = MagicMock(
            return_value="outputs/audio/test.mp3"
        )
        orchestrator.chart_generator.generate_trend_chart = MagicMock(
            return_value="data/charts/test.png"
        )
        orchestrator.video_assembler.assemble_video = MagicMock(
            return_value="outputs/videos/test.mp4"
        )
        
        print("   ✓ All components mocked")
        
        print("\n3. Running Pipeline for Test Date...")
        result = orchestrator.run_for_date("2024-07-04", "Yankees")
        
        if result:
            print(f"   ✓ Pipeline executed successfully")
            print(f"   ✓ Output: {result}")
        else:
            print("   ✗ Pipeline execution failed")
            return False
        
        print("\n4. Verifying Component Calls...")
        
        # Verify each component was called
        assert mock_fetcher.get_schedule.called, "Fetcher not called"
        print("   ✓ Data fetcher called")
        
        assert orchestrator.series_tracker.get_video_type.called, "Series tracker not called"
        print("   ✓ Series tracker called")
        
        assert orchestrator.analyzer.analyze_game.called, "Analyzer not called"
        print("   ✓ Analyzer called")
        
        assert orchestrator.script_generator.generate_script.called, "Script generator not called"
        print("   ✓ Script generator called")
        
        assert orchestrator.audio_generator.generate_audio.called, "Audio generator not called"
        print("   ✓ Audio generator called")
        
        assert orchestrator.video_assembler.assemble_video.called, "Video assembler not called"
        print("   ✓ Video assembler called")
        
        print("\n" + "="*60)
        print("✅ Phase 5 Verification PASSED!")
        print("="*60)
        print("\nThe pipeline orchestration is working correctly.")
        print("All components are properly integrated.")
        print("\nNext steps:")
        print("  1. Test with real data: python main.py --date YYYY-MM-DD --team Yankees")
        print("  2. Set up YouTube credentials for upload functionality")
        print("  3. Schedule daily runs via cron or GitHub Actions")
        
        return True

if __name__ == "__main__":
    success = verify_phase5()
    sys.exit(0 if success else 1)
