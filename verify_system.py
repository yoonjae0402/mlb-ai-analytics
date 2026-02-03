import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath("."))

from src.watcher import GameWatcher
from src.content.script_generator import ScriptGenerator
from src.content.image_generator import ImageGenerator
from src.video.cinematic_engine import CinematicEngine
from src.pipeline import PipelineOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

def verify_components():
    logger.info("Verifying Cinematic Pipeline Components...")

    # 1. Verify Watcher
    try:
        watcher = GameWatcher(poll_interval=10)
        logger.info("✅ GameWatcher initialized")
    except Exception as e:
        logger.error(f"❌ GameWatcher init failed: {e}")
        return

    # 2. Verify Script Generator
    try:
        script_gen = ScriptGenerator()
        # Mock Gemini client
        script_gen.client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '''
        {
          "video_metadata": {"title": "Test Video", "tags": [], "description": "desc", "duration_target": 30},
          "scenes": [
            {"scene_id": 1, "narration": "Hello world", "image_prompt": "Baseball scene", "motion_type": "zoom_in"}
          ]
        }
        '''
        script_gen.client.models.generate_content.return_value = mock_response
        
        # Test generation
        script = script_gen.generate_cinematic_script(
            game_data={"away_team": "A", "home_team": "B", "date": "2024-01-01"},
            analysis={"insights": ["Good game"]},
            prediction={"prediction": "Win", "confidence": "High"},
            video_type="series_middle"
        )
        assert script["scenes"][0]["scene_id"] == 1
        logger.info("✅ ScriptGenerator verified (Mocked)")
    except Exception as e:
        logger.error(f"❌ ScriptGenerator failed: {e}")
        return

    # 3. Verify Image Generator
    try:
        img_gen = ImageGenerator(api_key="verify_key")
        # Mock API call
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.headers = {"Content-Type": "image/png"}
            mock_post.return_value.content = b"fake_image_bytes"
            
            path = img_gen.generate_image("Test Prompt")
            assert path is not None
        logger.info("✅ ImageGenerator verified (Mocked)")
    except Exception as e:
        logger.error(f"❌ ImageGenerator failed: {e}")
        return

    # 4. Verify Cinematic Engine
    try:
        engine = CinematicEngine()
        # Just check methods exist, don't render (requires libs)
        assert hasattr(engine, "render_video")
        logger.info("✅ CinematicEngine initialized")
    except Exception as e:
        logger.error(f"❌ CinematicEngine init failed: {e}")
        return

    # 5. Verify Pipeline Orchestrator Integration
    try:
        orchestrator = PipelineOrchestrator()
        assert hasattr(orchestrator, "run_cinematic_for_date")
        logger.info("✅ PipelineOrchestrator initialized")
    except Exception as e:
        logger.error(f"❌ PipelineOrchestrator init failed: {e}")
        return

    logger.info("🎉 ALL SYSTEMS GO! The codebase structure is valid.")

if __name__ == "__main__":
    verify_components()
