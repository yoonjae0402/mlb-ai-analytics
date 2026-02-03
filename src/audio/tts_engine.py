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
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None

from config.settings import settings
from src.utils.logger import get_logger
from src.utils.cost_tracker import get_cost_tracker
from src.audio.google_tts import GoogleTTSGenerator


logger = get_logger(__name__)


class TTSEngine:
    """
    Text-to-speech engine using Google Cloud TTS with local Qwen3-TTS fallback.

    Generates natural-sounding narration for video scripts.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  # Smaller model for better compatibility
    DEFAULT_VOICE = "eric"

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
        self.model_id = model or self.DEFAULT_MODEL_ID
        self.voice = voice or settings.tts_voice or self.DEFAULT_VOICE
        self.output_dir = output_dir or settings.audio_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Initialize Google TTS
        self.google_tts = GoogleTTSGenerator()
        
        # Lazy load Qwen model only if needed? 
        # For now, we load it if we don't have Google credentials or ensuring fallback availability
        # But loading it takes VRAM. 
        # We will attempt to load it only on fallback OR if Google is not configured?
        # Requirement says "Add fallback to Qwen3-TTS". Implies it should be available.
        # But if we want 15x speed update, maybe don't load Qwen unless Google fails?
        # Let's initialize self.model = None and load lazily.
        self.model = None

        logger.info(f"TTS Engine initialized. Google TTS configured: {self.google_tts.health_check()}")
        self.cost_tracker = get_cost_tracker()

    def _ensure_local_model(self):
        """Lazy load local model if needed."""
        if self.model is not None:
            return

        if Qwen3TTSModel is None:
            raise ImportError("qwen-tts package is not installed. Run `pip install qwen-tts`.")

        self.device = self._select_device(self.device)
        logger.info(f"Initializing Local Qwen3-TTS with model: {self.model_id} on {self.device}")
        self.model = self._load_model()
        
        # Verify voice support
        supported = self.model.get_supported_speakers()
        if supported and self.voice.lower() not in [s.lower() for s in supported]:
            logger.warning(f"Voice '{self.voice}' not found in supported speakers. Using default '{supported[0]}'")
            self.voice = supported[0]

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
        voices = [{"voice_id": settings.google_tts_voice, "name": "Google Neural2", "source": "Google Cloud"}]
        
        # Add local voices if valid
        local_voices = [
            {"voice_id": "aiden", "name": "Aiden", "source": "Local"},
            {"voice_id": "eric", "name": "Eric", "source": "Local"},
        ]
        voices.extend(local_voices)
        return voices

    def set_voice(self, voice_id: str) -> None:
        """Set the voice to use for generation."""
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
        
        Tries Google Cloud TTS first, then falls back to local Qwen3-TTS.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Generate output filename (used for local output if not cached)
        if output_name is None:
            text_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
            output_name = f"narration_{text_hash}"

        # ---------------------------------------------------------------------
        # 1. Try Google TTS
        # ---------------------------------------------------------------------
        try:
            # We assume Google TTS handles its own caching
            audio_path = self.google_tts.generate(text)
            if audio_path:
                logger.info(f"Narrated via Google TTS: {audio_path.name}")
                return audio_path
        except Exception as e:
            logger.warning(f"Google TTS attempt failed: {e}")

        # ---------------------------------------------------------------------
        # 2. Fallback to Local Qwen3-TTS
        # ---------------------------------------------------------------------
        logger.info("Falling back to Local Qwen3-TTS...")
        
        if settings.dry_run:
            logger.info("DRY RUN: Would generate audio locally")
            path = self.output_dir / f"{output_name}_dryrun.wav"
            path.touch()
            return path

        try:
            self._ensure_local_model()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{output_name}_{timestamp}.wav"

            chunks = self._split_text(text)
            all_audio = []
            sample_rate = None

            for i, chunk in enumerate(chunks):
                wavs, sr = self.model.generate_custom_voice(
                    text=chunk,
                    speaker=self.voice,
                    non_streaming_mode=True
                )

                if wavs and len(wavs) > 0:
                    if sample_rate is None:
                        sample_rate = sr
                    all_audio.append(wavs[0])
                
                # Cleanup
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()

            if all_audio:
                combined_audio = np.concatenate(all_audio)
                wavfile.write(str(output_path), sample_rate, combined_audio)
                logger.info(f"Audio saved to {output_path} (Local)")
                return output_path
            else:
                 raise RuntimeError("No audio data generated from chunks")

        except Exception as e:
            logger.error(f"Failed to generate narration (Local): {e}")
            raise

    def _split_text(self, text: str, max_chars: int = 250) -> list[str]:
        """Split text into chunks at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            
            if current_length + len(sentence) > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

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

    # =========================================================================
    # Audio Processing
    # =========================================================================

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of an audio file in seconds."""
        try:
            # Try wave first
            try:
                import wave
                with wave.open(str(audio_path), 'rb') as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    return frames / float(rate)
            except:
                # Fallback to moviepy or other header reading for MP3 using mutagen if available
                # or just use ffmpeg via moviepy
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

    def health_check(self) -> bool:
        """Check if any TTS is available."""
        return self.google_tts.health_check() or (hasattr(self, 'model') and self.model is not None)
