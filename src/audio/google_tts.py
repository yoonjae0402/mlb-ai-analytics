
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

from google.cloud import texttospeech
from google.api_core import exceptions as google_exceptions

from config.settings import settings
from src.utils.logger import get_logger
from src.utils.cost_tracker import get_cost_tracker
from src.audio.tts_cache import TTSCache

logger = get_logger(__name__)


class GoogleTTSGenerator:
    """
    Google Cloud Text-to-Speech Generator.
    
    Features:
    - High-quality Neural2 input
    - Automatic caching
    - Cost tracking
    - Robust error handling
    """

    def __init__(self):
        self.client = None
        self.cache = TTSCache()
        self.cost_tracker = get_cost_tracker()
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Google Cloud TTS client."""
        try:
            # Set credentials env var if provided in settings
            if settings.google_application_credentials:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
            
            self.client = texttospeech.TextToSpeechClient()
            logger.info("Google Cloud TTS client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Google TTS client: {e}")
            self.client = None

    def generate(self, text: str, voice_name: Optional[str] = None) -> Optional[Path]:
        """
        Generate audio from text using Google Cloud TTS.
        
        Args:
            text: Text to synthesize.
            voice_name: Voice ID (default: from settings).
            
        Returns:
            Path to generated audio file, or None if generation failed.
        """
        if not text:
            return None
            
        voice_name = voice_name or settings.google_tts_voice
        model_name = "google-tts" # for cache key
        
        # 1. Check Cache
        cached_path = self.cache.get(text, voice_name, model_name)
        if cached_path:
            return cached_path
            
        # 2. Check Client
        if not self.client:
            logger.warning("Google TTS client not available")
            return None
            
        # 3. Check Daily Limit (heuristic)
        try:
            daily_cost = self.cost_tracker.get_daily_cost()
            estimated_cost = self.cost_tracker.calculate_tts_cost(text, "neural2")
            if daily_cost + estimated_cost > settings.daily_cost_limit: 
                # This checks total budget, but we also have a char limit setting
                logger.warning(f"Daily cost limit reached (${daily_cost:.2f}). Skipping Google TTS.")
                return None
        except Exception as e:
            logger.warning(f"Cost check failed: {e}")

        # 4. Generate
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Build voice params
            language_code = "-".join(voice_name.split("-")[:2]) # e.g. en-US from en-US-Neural2-J
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            # Select audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                # Optional: pitch=0.0, volume_gain_db=0.0
            )

            logger.info(f"Calling Google TTS API ({len(text)} chars, {voice_name})")
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )

            # 5. Save and Cache
            audio_content = response.audio_content
            saved_path = self.cache.save(
                audio_content, 
                text, 
                voice_name, 
                model_name, 
                extension="mp3"
            )
            
            # 6. Track Cost
            cost = self.cost_tracker.calculate_tts_cost(text, "neural2" if "Neural2" in voice_name else "standard")
            self.cost_tracker.track_cost("google_tts", cost, f"TTS: {len(text)} chars")
            
            return saved_path

        except google_exceptions.GoogleAPICallError as e:
            logger.error(f"Google TTS API call failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Google TTS generation: {e}")
            return None

    def health_check(self) -> bool:
        """Check if client is initialized and valid."""
        return self.client is not None
