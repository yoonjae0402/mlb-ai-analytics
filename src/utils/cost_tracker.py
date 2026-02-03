"""
Cost tracker for monitoring API usage across providers.
Tracks costs for Gemini, Nano Banana, and other services.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Tracks API costs per provider and logs them to a JSONL file.
    """

    def __init__(self, log_file: str = "data/costs.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._session_costs: Dict[str, float] = {}

    def track_cost(
        self,
        provider: str,
        cost: float,
        details: str = "",
    ) -> None:
        """
        Log a cost event.

        Args:
            provider: Service name (e.g. 'gemini', 'nano_banana', 'tts').
            cost: Cost in USD.
            details: Optional description of what was generated.
        """
        self._session_costs[provider] = self._session_costs.get(provider, 0) + cost

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "cost": cost,
            "details": details,
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning(f"Failed to write cost entry: {e}")

        if cost > 0:
            logger.debug(f"Cost tracked: {provider} ${cost:.4f} ({details})")

    def get_session_cost(self, provider: Optional[str] = None) -> float:
        """Get accumulated cost for this session, optionally filtered by provider."""
        if provider:
            return self._session_costs.get(provider, 0.0)
        return sum(self._session_costs.values())

    def get_total_cost(self) -> float:
        """Returns total cost across all providers for this session."""
        return self.get_session_cost()

    def get_daily_cost(self) -> float:
        """Sum all costs from today's entries in the JSONL file."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        total = 0.0

        if not self.log_file.exists():
            return total

        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("timestamp", "").startswith(today):
                            total += entry.get("cost", 0.0)
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning(f"Failed to read cost log: {e}")

        return total

    def check_budget(self, threshold: Optional[float] = None) -> bool:
        """
        Check if daily costs are within budget.

        Args:
            threshold: Max daily spend in USD. Defaults to 10.0.

        Returns:
            True if within budget, False if exceeded.
        """
        limit = threshold if threshold is not None else 10.0
        daily = self.get_daily_cost()
        within = daily <= limit
        if not within:
            logger.warning(f"Daily budget exceeded: ${daily:.2f} / ${limit:.2f}")
        return within

    GOOGLE_TTS_PRICES = {
        "standard": 0.000004,  # $4.00 per 1M chars
        "neural2": 0.000016,   # $16.00 per 1M chars
        "studio": 0.000160,    # $160.00 per 1M chars
    }

    def calculate_tts_cost(self, text: str, voice_type: str = "neural2") -> float:
        """
        Calculate estimated cost for Google TTS.
        
        Args:
            text: Input text
            voice_type: 'standard', 'neural2', or 'studio'
            
        Returns:
            Estimated cost in USD
        """
        char_count = len(text)
        price_per_char = self.GOOGLE_TTS_PRICES.get(voice_type.lower(), 0.000016)
        return char_count * price_per_char

    def get_monthly_cost(self) -> float:
        """Sum all costs from this month's entries."""
        this_month = datetime.now(timezone.utc).strftime("%Y-%m")
        total = 0.0
        
        if not self.log_file.exists():
            return total

        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("timestamp", "").startswith(this_month):
                            total += entry.get("cost", 0.0)
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning(f"Failed to read cost log for monthly total: {e}")

        return total
        
    def export_csv(self, output_path: str = "costs_report.csv") -> None:
        """Export cost logs to CSV."""
        import csv
        
        if not self.log_file.exists():
            return
            
        try:
            entries = []
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except:
                        continue
                        
            if not entries:
                return
                
            keys = ["timestamp", "provider", "cost", "details"]
            
            with open(output_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                for entry in entries:
                    # Filter to only known keys to avoid errors if extra data exists
                    row = {k: entry.get(k, "") for k in keys}
                    writer.writerow(row)
                    
            logger.info(f"Cost report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")

def get_cost_tracker() -> CostTracker:
    """Get cost tracker instance."""
    return CostTracker()
