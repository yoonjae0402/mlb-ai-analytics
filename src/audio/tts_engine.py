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

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None

from config.settings import settings
from src.utils.logger import get_logger
from src.utils.cost_tracker import get_cost_tracker


logger = get_logger(__name__)


class TTSEngine:
    """
    Text-to-speech engine using local Qwen3-TTS.

    Generates natural-sounding narration for video scripts.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  # Smaller model for better compatibility
    DEFAULT_VOICE = "aiden"

    def __init__(
        self,
        api_key: str | None = None, # Kept for compatibility, unused
        model: str | None = None,
        voice: str | None = None,
        output_dir: Path | None = None,
        device: str | None = None
    ):
        """
        Initialize the TTS engine.

        Args:
            api_key: Unused (kept for compatibility)
            model: Model ID to use (default: Qwen3-TTS 0.6B CustomVoice)
            voice: Voice ID to use
            output_dir: Directory for audio output
            device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
        """
        if Qwen3TTSModel is None:
            raise ImportError("qwen-tts package is not installed. Run `pip install qwen-tts`.")

        self.model_id = model or self.DEFAULT_MODEL_ID
        self.voice = voice or settings.tts_voice or self.DEFAULT_VOICE
        self.output_dir = output_dir or settings.audio_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device selection: prefer CUDA > MPS > CPU
        self.device = self._select_device(device)

        logger.info(f"Initializing Qwen3-TTS with model: {self.model_id} on {self.device}")

        self.model = self._load_model()

        # Verify voice support
        supported = self.model.get_supported_speakers()
        if supported:
            logger.info(f"Supported voices: {supported}")
            if self.voice.lower() not in [s.lower() for s in supported]:
                logger.warning(f"Voice '{self.voice}' not found in supported speakers. Using default '{supported[0]}'")
                self.voice = supported[0]

        logger.info(f"Qwen3-TTS model loaded successfully on {self.device}")

        self.cost_tracker = get_cost_tracker()

    # =========================================================================
    # Device & Model Helpers
    # =========================================================================

    @staticmethod
    def _select_device(override: str | None) -> str:
        """Pick the best available accelerator."""
        if override:
            return override
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load the TTS model, falling back to CPU if GPU fails."""
        try:
            model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
            )
            return model
        except Exception as e:
            if self.device != "cpu":
                logger.warning(f"Failed to load model on {self.device}: {e}. Falling back to CPU.")
                self.device = "cpu"
                return Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    device_map="cpu",
                )
            raise

    # =========================================================================
    # Voice Management
    # =========================================================================

    def list_voices(self) -> list[dict[str, Any]]:
        """
        List available voices from the model.
        """
        if hasattr(self, 'model'):
            speakers = self.model.get_supported_speakers()
            if speakers:
                return [{"voice_id": s, "name": s, "language": "Multilingual"} for s in speakers]
        
        # Fallback if model not loaded or no speakers listed
        return [
            {"voice_id": "aiden", "name": "Aiden", "language": "Multilingual"},
            {"voice_id": "dylan", "name": "Dylan", "language": "Multilingual"},
            {"voice_id": "eric", "name": "Eric", "language": "Multilingual"},
            {"voice_id": "ryan", "name": "Ryan", "language": "Multilingual"},
            {"voice_id": "serena", "name": "Serena", "language": "Multilingual"},
            {"voice_id": "vivian", "name": "Vivian", "language": "Multilingual"},
        ]

    def set_voice(self, voice_id: str) -> None:
        """
        Set the voice to use for generation.

        Args:
            voice_id: Voice ID (speaker name)
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
        Generate audio narration from text using local Qwen3-TTS.

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
        output_path = self.output_dir / f"{output_name}_{timestamp}.wav" # Using wav for direct output

        logger.info(f"Generating narration: {len(text)} characters using voice '{self.voice}'")

        if settings.dry_run:
            logger.info("DRY RUN: Would generate audio")
            output_path.touch()
            return output_path

        try:
            # Generate audio using Qwen3-TTS
            # generate_custom_voice returns (wavs, sample_rate)
            # wavs is a list of numpy arrays (one per input text)
            wavs, sample_rate = self.model.generate_custom_voice(
                text=text,
                speaker=self.voice,
                non_streaming_mode=True
            )

            if wavs and len(wavs) > 0:
                audio_data = wavs[0]
                
                # Normalize float32 audio to int16 if necessary or save as is
                # scipy.io.wavfile.write handles float32 [-1.0, 1.0] usually
                
                wavfile.write(str(output_path), sample_rate, audio_data)
                
                logger.info(f"Audio saved to {output_path}")
                return output_path
            else:
                 raise RuntimeError("No audio data generated")

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
            # Use scipy to get duration directly if moviepy fails or to reduce dependencies
            try:
                rate, data = wavfile.read(str(audio_path))
                return len(data) / rate
            except:
                from moviepy.editor import AudioFileClip
                with AudioFileClip(str(audio_path)) as audio:
                    return audio.duration
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0

    def estimate_duration(self, text: str, wpm: int = 150) -> float:
        """Estimate narration duration."""
        word_count = len(text.split())
        return (word_count / wpm) * 60

    def estimate_cost(self, text: str) -> float:
        """Estimate cost."""
        # Local inference is free!
        return 0.0

    def health_check(self) -> bool:
        """
        Check if the model is loaded.
        """
        return hasattr(self, 'model') and self.model is not None
