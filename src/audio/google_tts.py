
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

try:
    from google.cloud import texttospeech
    from google.api_core import exceptions as google_exceptions
    _GOOGLE_TTS_AVAILABLE = True
except ImportError:
    texttospeech = None
    google_exceptions = None
    _GOOGLE_TTS_AVAILABLE = False

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
        if not _GOOGLE_TTS_AVAILABLE:
            logger.warning("google-cloud-texttospeech package not installed. Google TTS unavailable.")
            self.client = None
            return

        try:
            # Set credentials env var if provided in settings
            if settings.google_application_credentials:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials

            self.client = texttospeech.TextToSpeechClient()
            logger.info("Google Cloud TTS client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Google TTS client: {e}")
            self.client = None

    def generate(self, text: str, voice_name: Optional[str] = None, speaking_rate: float = 1.0, ssml: Optional[str] = None) -> Optional[Path]:
        """
        Generate audio from text using Google Cloud TTS.

        Args:
            text: Text to synthesize.
            voice_name: Voice ID (default: from settings).
            speaking_rate: Speed of speech (0.25 to 4.0). 1.0 is normal.
            ssml: Optional SSML markup to use instead of plain text.

        Returns:
            Path to generated audio file, or None if generation failed.
        """
        if not text and not ssml:
            return None

        voice_name = voice_name or settings.google_tts_voice
        model_name = "google-tts"

        cache_text = ssml or text
        voice_key = f"{voice_name}_rate_{speaking_rate}"

        cached_path = self.cache.get(cache_text, voice_key, model_name)
        if cached_path:
            return cached_path

        if not self.client:
            logger.warning("Google TTS client not available")
            return None

        try:
            daily_cost = self.cost_tracker.get_daily_cost()
            estimated_cost = self.cost_tracker.calculate_tts_cost(text or ssml, "neural2")
            if daily_cost + estimated_cost > settings.daily_cost_limit:
                logger.warning(f"Daily cost limit reached (${daily_cost:.2f}). Skipping Google TTS.")
                return None
        except Exception as e:
            logger.warning(f"Cost check failed: {e}")

        try:
            if ssml:
                synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
            else:
                synthesis_input = texttospeech.SynthesisInput(text=text)

            language_code = "-".join(voice_name.split("-")[:2])
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
            )

            logger.info(f"Calling Google TTS API ({len(text)} chars, {voice_name}, rate={speaking_rate})")
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
                voice_key, 
                model_name, 
                extension="mp3"
            )
            
            # 6. Track Cost
            cost = self.cost_tracker.calculate_tts_cost(text, "neural2" if "Neural2" in voice_name else "standard")
            self.cost_tracker.track_cost("google_tts", cost, f"TTS: {len(text)} chars")
            
            return saved_path

        except Exception as e:
            if google_exceptions and isinstance(e, google_exceptions.GoogleAPICallError):
                logger.error(f"Google TTS API call failed: {e}")
                return None
            logger.error(f"Unexpected error in Google TTS generation: {e}")
            return None

    def health_check(self) -> bool:
        """Check if client is initialized and valid."""
        return self.client is not None
