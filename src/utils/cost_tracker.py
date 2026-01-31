import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

class CostTracker:
    """
    Tracks API costs (OpenAI, DashScope) to a local file.
    """
    
    def __init__(self, log_file: str = "data/costs.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Approximate costs (as of Jan 2025)
        self.PRICING = {
            "gpt-4o-input": 2.50 / 1e6, # per token
            "gpt-4o-output": 10.00 / 1e6, # per token
            "dashscope": 2.00 / 1e6, # per character ($2 per 1M chars approx)
        }
        
    def log_openai_usage(
        self, 
        model: str, 
        input_tokens: int, 
        output_tokens: int
    ):
        """Log OpenAI API usage."""
        input_cost = input_tokens * self.PRICING.get(f"{model}-input", 0)
        output_cost = output_tokens * self.PRICING.get(f"{model}-output", 0)
        total_cost = input_cost + output_cost
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": "openai",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": float(f"{total_cost:.5f}")
        }
        self._write_entry(entry)
        
    def log_dashscope_usage(self, model: str, characters: int):
        """Log DashScope API usage."""
        cost = characters * self.PRICING["dashscope"]
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": "dashscope",
            "model": model,
            "characters": characters,
            "cost_usd": float(f"{cost:.5f}")
        }
        self._write_entry(entry)

    def _write_entry(self, entry: dict):
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write cost log: {e}")
            
    def get_total_cost(self) -> float:
        """Calculate total project cost."""
        total = 0.0
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        total += entry.get("cost_usd", 0)
                    except:
                        pass
        return total

@lru_cache()
def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance."""
    return CostTracker()
