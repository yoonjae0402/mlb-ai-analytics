
import logging
from config.viral_video_config import ViralVideoConfig
from src.utils.error_handler import ConfigurationError

logger = logging.getLogger(__name__)

class BudgetExceededError(Exception):
    pass

class TTSCostTracker:
    """
    Tracks Google Cloud TTS usage and costs.
    Pricing (Neural2): ~$16.00 per 1 million characters.
    """
    
    NEURAL2_COST_PER_MILLION = 16.00
    DAILY_LIMIT_USD = 5.00 # Safety limit
    
    _daily_chars = 0
    _daily_cost = 0.0
    
    @classmethod
    def track_generation(cls, text: str):
        """Log usage for a request."""
        chars = len(text)
        cost = (chars / 1_000_000) * cls.NEURAL2_COST_PER_MILLION
        
        cls._daily_chars += chars
        cls._daily_cost += cost
        
        logger.info(f"TTS Cost: ${cost:.5f} | Daily Total: ${cls._daily_cost:.2f} | Chars: {chars}")
        
        if cls._daily_cost > cls.DAILY_LIMIT_USD:
            msg = f"Daily TTS cost ${cls._daily_cost:.2f} exceeds limit ${cls.DAILY_LIMIT_USD}"
            logger.error(msg)
            raise BudgetExceededError(msg)
            
    @classmethod
    def get_usage(cls):
        return {
            "chars": cls._daily_chars,
            "cost_usd": cls._daily_cost
        }
