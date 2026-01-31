import logging
import os
from typing import Optional
from pathlib import Path

from src.audio.tts_engine import TTSEngine
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

class AudioGenerator:
    """
    Generates audio from text using TTSEngine (DashScope).
    """
    
    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        try:
            self.engine = TTSEngine()
        except Exception as e:
            logger.warning(f"TTSEngine initialization failed: {e}")
            self.engine = None
            
        self.cost_tracker = cost_tracker or CostTracker()
        
    def generate_audio(self, text: str, output_path: str) -> Optional[str]:
        """
        Synthesize speech from text.
        """
        if not self.engine:
            logger.error("TTS Engine not initialized.")
            return None
            
        try:
            logger.info(f"Generating audio for: '{text[:30]}...'")
            
            # Generate using TTSEngine
            # TTSEngine generates a hash-based filename by default, so we might need to move it
            # or just use TTSEngine's logic.
            # But the signature here takes specific output_path.
            
            # Since TTSEngine.generate_narration returns a Path, 
            # and handles file creation, let's adapt.
            
            # If output_path is provided, we might want to respect it.
            # TTSEngine.generate_narration takes output_name (basename).
            
            target_path = Path(output_path)
            output_name = target_path.stem
            
            # Temporarily override output_dir of engine if needed, or just move file after.
            # Cleaner: Update TTSEngine to accept absolute path or just use it as is if straightforward.
            # Let's just blindly use engine and move the file for now, or use engine's output_dir if it matches.
            
            generated_path = self.engine.generate_narration(text, output_name=output_name)
            
            # If the generated path is not the target path, move it?
            # TTSEngine appends timestamp.
            
            if generated_path != target_path:
                # Ensuring target directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Rename/Move
                generated_path.replace(target_path)
                logger.info(f"Moved audio to {target_path}")
                return str(target_path)

            return str(generated_path)
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None
