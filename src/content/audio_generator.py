import logging
import os
from typing import Optional
from pathlib import Path

from src.audio.tts_engine import TTSEngine
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

# Shorts videos should be under 60s.  At ~150 WPM, 80 words ≈ 32s of speech.
MAX_SCRIPT_WORDS = 80


class AudioGenerator:
    """
    Generates audio from text using TTSEngine (Google Cloud TTS with Qwen3-TTS fallback).
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        try:
            self.engine = TTSEngine()
        except Exception as e:
            logger.warning(f"TTSEngine initialization failed: {e}")
            self.engine = None

        self.cost_tracker = cost_tracker or CostTracker()

    @staticmethod
    def _truncate(text: str, max_words: int = MAX_SCRIPT_WORDS) -> str:
        """Hard-cap the script to *max_words* words so TTS stays short."""
        words = text.split()
        if len(words) <= max_words:
            return text
        logger.warning(f"Script too long ({len(words)} words), truncating to {max_words}")
        return " ".join(words[:max_words])

    def generate_audio(self, text: str, output_path: str, speaking_rate: float = 1.0) -> Optional[str]:
        """
        Synthesize speech from text.
        """
        if not self.engine:
            logger.error("TTS Engine not initialized.")
            return None

        try:
            text = self._truncate(text)
            logger.info(f"Generating audio for: '{text[:30]}...' ({len(text.split())} words, rate={speaking_rate})")
            
            target_path = Path(output_path)
            output_name = target_path.stem
            
            # Pass speaking_rate to generate_narration
            # Note: We need to verify TTSEngine supports speaking_rate too.
            generated_path = self.engine.generate_narration(
                text, 
                output_name=output_name,
                speaking_rate=speaking_rate
            )
            
            if generated_path != target_path:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                generated_path.replace(target_path)
                logger.debug(f"Moved audio to {target_path}")
                return str(target_path)

            return str(generated_path)
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None
