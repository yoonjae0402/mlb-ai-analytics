"""
MLB Video Pipeline - Text-to-Speech Engine

Generates natural narration using local Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
Runs entirely locally without API keys.
"""

from pathlib import Path
from typing import Any
from datetime import datetime
import hashlib
import torch
import scipy.io.wavfile as wavfile
import numpy as np
import re
import gc



try:
    # Qwen import removed
    pass
except ImportError:
    pass

from config.settings import settings
from src.utils.logger import get_logger
from src.utils.cost_tracker import get_cost_tracker
from src.audio.google_tts import GoogleTTSGenerator


logger = get_logger(__name__)


class TTSEngine:
    """
    Text-to-speech engine using Google Cloud TTS.
    
    Now simplified to remove local Qwen3-TTS fallback. 
    Requires Google Cloud Credentials.
    """

    DEFAULT_VOICE = "en-US-Neural2-J"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        output_dir: Path | None = None,
        device: str | None = None
    ):
        """
        Initialize the TTS engine.
        """
        self.voice = voice or settings.tts_voice or self.DEFAULT_VOICE
        self.output_dir = output_dir or settings.audio_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Google TTS
        self.google_tts = GoogleTTSGenerator()
        
        logger.info(f"TTS Engine initialized. Google TTS configured: {self.google_tts.health_check()}")
        self.cost_tracker = get_cost_tracker()

    def list_voices(self) -> list[dict[str, Any]]:
        """
        List available voices.
        """
        return [{"voice_id": settings.google_tts_voice, "name": "Google Neural2", "source": "Google Cloud"}]

    def set_voice(self, voice_id: str) -> None:
        """Set the voice to use for generation."""
        self.voice = voice_id
        logger.info(f"Voice set to: {voice_id}")

    def generate_narration(
        self,
        text: str,
        output_name: str | None = None,
    ) -> Path:
        """
        Generate audio narration from text using Google Cloud TTS.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # 1. Try Google TTS
        try:
            # We assume Google TTS handles its own caching
            # Pass the voice ID if it differs from default
            audio_path = self.google_tts.generate(text, voice_name=self.voice)
            if audio_path:
                logger.info(f"Narrated via Google TTS: {audio_path.name}")
                return audio_path
            else:
                raise RuntimeError("Google TTS returned no path.")
        except Exception as e:
            logger.error(f"Google TTS generation failed: {e}")
            raise RuntimeError(f"TTS Failed: {e}. Please check GOOGLE_APPLICATION_CREDENTIALS.")

    def generate_segments(
        self,
        segments: list[dict[str, str]],
        output_prefix: str = "segment",
    ) -> list[Path]:
        """Generate multiple audio segments."""
        audio_paths = []

        for i, segment in enumerate(segments):
            text = segment.get("text", "")
            output_name = f"{output_prefix}_{i:03d}"
            try:
                path = self.generate_narration(text, output_name)
                audio_paths.append(path)
            except Exception as e:
                logger.error(f"Failed segment {i}: {e}")
                # Continue best effort?
                continue

        return audio_paths

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of an audio file in seconds."""
        try:
            # Try wave first if it's wav (Google might return mp3 though)
            try:
                import wave
                with wave.open(str(audio_path), 'rb') as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    return frames / float(rate)
            except:
                # Fallback to moviepy or other header reading
                from moviepy.editor import AudioFileClip
                try:
                    with AudioFileClip(str(audio_path)) as audio:
                        return audio.duration
                except OSError:
                    # Sometimes AudioFileClip needs ffmpeg and might fail if not found or file empty
                    logger.warning(f"Could not determine duration for {audio_path}")
                    return 0.0
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0

    def estimate_duration(self, text: str, wpm: int = 150) -> float:
        """Estimate narration duration."""
        word_count = len(text.split())
        return (word_count / wpm) * 60

    def health_check(self) -> bool:
        """Check if TTS is available."""
        return self.google_tts.health_check()
