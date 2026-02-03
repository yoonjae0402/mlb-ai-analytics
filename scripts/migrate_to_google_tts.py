#!/usr/bin/env python3
"""
Migration Script: Local Qwen3-TTS -> Google Cloud TTS
"""
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.google_tts import GoogleTTSGenerator
from src.audio.tts_engine import TTSEngine
from src.utils.cost_tracker import get_cost_tracker
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

def test_google_integration():
    logger.info("="*50)
    logger.info("TESTING GOOGLE CLOUD TTS INTEGRATION")
    logger.info("="*50)
    
    if not settings.google_application_credentials:
        logger.warning("No Google Credentials found in settings!")
        logger.info("Please set GOOGLE_APPLICATION_CREDENTIALS in .env")
        return False
        
    generator = GoogleTTSGenerator()
    if not generator.health_check():
        logger.error("Google TTS Client failed to initialize.")
        return False
        
    text = "This is a test of the Google Cloud Text-to-Speech system for the MLB Video Pipeline."
    
    # Test 1: Generation
    start = time.time()
    path = generator.generate(text)
    duration = time.time() - start
    
    if path and path.exists():
        logger.info(f"[SUCCESS] Generated audio in {duration:.2f}s")
        logger.info(f"File: {path}")
    else:
        logger.error("[FAILED] Audio generation returned None")
        return False

    # Test 2: Caching
    logger.info("\nTesting Cache...")
    start = time.time()
    path_cached = generator.generate(text)
    duration_cached = time.time() - start
    
    if path_cached == path:
        logger.info(f"[SUCCESS] Cache hit confirmed! ({duration_cached:.4f}s)")
    else:
         logger.warning(f"[WARNING] Cache miss or different file? {path_cached} vs {path}")

    # Test 3: Cost
    tracker = get_cost_tracker()
    daily_cost = tracker.get_daily_cost()
    logger.info(f"\nDaily Cost so far: ${daily_cost:.4f}")
    
    return True

def test_full_pipeline_fallback():
    logger.info("\n" + "="*50)
    logger.info("TESTING PIPELINE FALLBACK")
    logger.info("="*50)
    
    engine = TTSEngine()
    
    text = "Transitioning to the next inning."
    
    # This should use Google if configured, otherwise Qwen
    path = engine.generate_narration(text)
    
    if path and path.exists():
        logger.info(f"[SUCCESS] Engine generated: {path.name}")
        # Verify if it was Google (mp3) or Qwen (wav) depending on extension usually, 
        # but our code saves google as mp3 in cache, but TTSEngine might use it directly.
        # Check logs for "Narrated via Google TTS"
    else:
        logger.error("[FAILED] Engine failed to generate audio")

if __name__ == "__main__":
    success = test_google_integration()
    if success:
        test_full_pipeline_fallback()
    else:
        logger.error("Skipping fallback test due to Google integration failure")
