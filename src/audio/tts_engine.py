"""
MLB Video Pipeline - Text-to-Speech Engine

Generates natural narration using Alibaba Cloud DashScope (Qwen3-TTS).
Handles voice selection, audio generation, and file management.

Compatible with DashScope SDK
"""

from pathlib import Path
from typing import Any
from datetime import datetime
import hashlib
import json

import dashscope
from dashscope.audio.tts import SpeechSynthesizer

from config.settings import settings
from src.utils.logger import get_logger
from src.utils.cost_tracker import get_cost_tracker


logger = get_logger(__name__)


class TTSEngine:
    """
    Text-to-speech engine using DashScope (Qwen3-TTS).

    Generates natural-sounding narration for video scripts.
    """

    # DashScope Qwen3 pricing (approximate, per 1K characters)
    # Adjust based on settings
    PRICE_PER_1K_CHARS = settings.dashscope_cost_per_million_chars / 1000

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize the TTS engine.

        Args:
            api_key: DashScope API key
            model: Model ID to use
            voice: Voice ID to use
            output_dir: Directory for audio output
        """
        self.api_key = api_key or settings.dashscope_api_key
        if not self.api_key:
            raise ValueError("DashScope API key not configured")
        
        # Configure dashscope
        dashscope.api_key = self.api_key

        self.model = model or settings.tts_model
        self.voice = voice or settings.tts_voice
        self.output_dir = output_dir or settings.audio_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cost_tracker = get_cost_tracker()

        logger.info(f"TTSEngine initialized with model: {self.model}, voice: {self.voice}")

    # =========================================================================
    # Voice Management
    # =========================================================================

    def list_voices(self) -> list[dict[str, Any]]:
        """
        List available voices.
        
        DashScope currently has a set list of voices for Qwen3-TTS/CosyVoice.
        Returning a static list of popular ones for now.
        """
        return [
            {"voice_id": "longxiaochun", "name": "Long Xiaochun (Female)", "language": "Chinese/English"},
            {"voice_id": "longwan", "name": "Long Wan (Male)", "language": "Chinese/English"},
            {"voice_id": "longcheng", "name": "Long Cheng (Male)", "language": "Chinese/English"},
            {"voice_id": "longhua", "name": "Long Hua (Female)", "language": "Chinese/English"},
        ]

    def set_voice(self, voice_id: str) -> None:
        """
        Set the voice to use for generation.

        Args:
            voice_id: DashScope voice ID
        """
        self.voice = voice_id
        logger.info(f"Voice set to: {voice_id}")

    # =========================================================================
    # Audio Generation
    # =========================================================================

    def generate_narration(
        self,
        text: str,
        output_name: str | None = None,
    ) -> Path:
        """
        Generate audio narration from text.

        Args:
            text: Script text to convert to speech
            output_name: Base name for output file

        Returns:
            Path to generated audio file
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Generate output filename
        if output_name is None:
            text_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
            output_name = f"narration_{text_hash}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{output_name}_{timestamp}.mp3"

        logger.info(f"Generating narration: {len(text)} characters")

        if settings.dry_run:
            logger.info("DRY RUN: Would generate audio")
            output_path.touch()
            return output_path

        try:
            # Generate audio using DashScope
            result = SpeechSynthesizer.call(
                model=self.model,
                text=text,
                voice=self.voice,
                format='mp3'
            )

            if result.get_audio_data() is not None:
                with open(output_path, 'wb') as f:
                    f.write(result.get_audio_data())
            else:
                 logger.error(f"DashScope API Error: {result}")
                 raise RuntimeError(f"Failed to generate audio: {result.message}")

            # Track cost
            char_count = len(text)
            
            # Log usage
            if hasattr(self.cost_tracker, 'log_dashscope_usage'):
                self.cost_tracker.log_dashscope_usage(self.model, char_count)
            else:
                logger.warning("Cost tracker does not support dashscope logging")

            logger.info(f"Audio saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate narration: {e}")
            raise

    def generate_segments(
        self,
        segments: list[dict[str, str]],
        output_prefix: str = "segment",
    ) -> list[Path]:
        """
        Generate multiple audio segments.

        Args:
            segments: List of {"text": str, "voice": str (optional)}
            output_prefix: Prefix for output filenames

        Returns:
            List of paths to generated audio files
        """
        audio_paths = []

        for i, segment in enumerate(segments):
            text = segment.get("text", "")
            voice = segment.get("voice", self.voice)

            # Temporarily change voice if specified
            original_voice = self.voice
            if voice != self.voice:
                self.voice = voice

            try:
                output_name = f"{output_prefix}_{i:03d}"
                path = self.generate_narration(text, output_name)
                audio_paths.append(path)
            finally:
                self.voice = original_voice

        logger.info(f"Generated {len(audio_paths)} audio segments")
        return audio_paths

    # =========================================================================
    # Audio Processing
    # =========================================================================

    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get duration of an audio file in seconds.
        """
        try:
            from moviepy.editor import AudioFileClip
            with AudioFileClip(str(audio_path)) as audio:
                return audio.duration
        except ImportError:
            logger.warning("moviepy not available, cannot get audio duration")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0

    def estimate_duration(self, text: str, wpm: int = 150) -> float:
        """Estimate narration duration."""
        word_count = len(text.split())
        return (word_count / wpm) * 60

    def estimate_cost(self, text: str) -> float:
        """Estimate cost."""
        char_count = len(text)
        return (char_count / 1000) * self.PRICE_PER_1K_CHARS

    def health_check(self) -> bool:
        """
        Check if the DashScope API is accessible.
        
        We'll just try to instantiate a synthesizer or do a weak check.
        API Key is already checked in init.
        """
        return bool(self.api_key)
