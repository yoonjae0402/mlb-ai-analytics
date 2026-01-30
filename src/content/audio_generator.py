import os
import logging
from typing import Optional
from pathlib import Path
from elevenlabs.client import ElevenLabs
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

class AudioGenerator:
    """
    Generates audio from text using ElevenLabs API.
    """
    
    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb") # Default to 'George' or similar
        
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not found.")
            self.client = None
        else:
            self.client = ElevenLabs(api_key=self.api_key)
            
        self.cost_tracker = cost_tracker or CostTracker()
        
    def generate_audio(self, text: str, output_path: str) -> Optional[str]:
        """
        Synthesize speech from text.
        """
        if not self.client:
            logger.error("ElevenLabs client not initialized.")
            return None
            
        try:
            logger.info(f"Generating audio for: '{text[:30]}...'")
            
            # Generate
            audio = self.client.generate(
                text=text,
                voice=self.voice_id,
                model="eleven_monolingual_v1"
            )
            
            # Save
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # audio is a generator, consume it to write
            with open(path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)
            
            # Track cost
            self.cost_tracker.log_elevenlabs_usage(self.voice_id, len(text))
            
            logger.info(f"Audio saved to {path}")
            return str(path)
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None
