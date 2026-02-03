"""
Simple cost tracker stub for backward compatibility.
Since we now use Gemini (variable cost) and local Qwen3-TTS (free),
detailed cost tracking is no longer needed.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Minimal cost tracker for backward compatibility.
    
    Note: With Gemini API and local TTS, costs are minimal and tracked
    via the MetricsCollector instead.
    """
    
    def __init__(self, log_file: str = "data/costs.jsonl"):
        self.log_file = Path(log_file)
        logger.debug("CostTracker initialized (legacy compatibility mode)")
        
    def get_total_cost(self) -> float:
        """Returns 0.0 - costs now tracked in MetricsCollector."""
        return 0.0


def get_cost_tracker() -> CostTracker:
    """Get cost tracker instance."""
    return CostTracker()
