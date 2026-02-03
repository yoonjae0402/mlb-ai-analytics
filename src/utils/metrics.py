import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Collects and stores pipeline execution metrics.
    """
    
    def __init__(self, metrics_file: str = "data/metrics.jsonl"):
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_pipeline_run(
        self,
        date: str,
        team: str,
        success: bool,
        duration: float,
        costs: Dict[str, float],
        error: Optional[str] = None
    ):
        """
        Log a pipeline execution.
        
        Args:
            date: Game date (YYYY-MM-DD)
            team: Team name
            success: Whether pipeline completed successfully
            duration: Execution time in seconds
            costs: Dictionary of costs (gemini, total)
            error: Error message if failed
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "date": date,
            "team": team,
            "success": success,
            "duration": duration,
            "costs": costs,
            "error": error
        }
        
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            logger.info(f"Logged pipeline run: {team} on {date}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def get_recent_runs(self, days: int = 7) -> List[Dict]:
        """
        Get pipeline runs from the last N days.
        """
        if not self.metrics_file.exists():
            return []
        
        cutoff = datetime.now() - timedelta(days=days)
        runs = []
        
        try:
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    if timestamp >= cutoff:
                        runs.append(entry)
            
            return sorted(runs, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            logger.error(f"Failed to read metrics: {e}")
            return []
    
    def get_cost_summary(self, start_date: str, end_date: str) -> Dict:
        """
        Calculate cost summary for a date range.
        
        Returns:
            Dictionary with total costs and breakdown
        """
        if not self.metrics_file.exists():
            return {"total": 0, "gemini": 0, "runs": 0}
        
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        total_gemini = 0
        run_count = 0
        
        try:
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    
                    if start <= timestamp <= end:
                        costs = entry.get('costs', {})
                        total_gemini += costs.get('gemini', 0)
                        run_count += 1
            
            return {
                "total": total_gemini,
                "gemini": total_gemini,
                "runs": run_count
            }
        except Exception as e:
            logger.error(f"Failed to calculate cost summary: {e}")
            return {"total": 0, "gemini": 0, "runs": 0}
    
    def get_success_rate(self, days: int = 7) -> float:
        """
        Calculate success rate for recent runs.
        """
        runs = self.get_recent_runs(days)
        if not runs:
            return 0.0
        
        successful = sum(1 for r in runs if r['success'])
        return (successful / len(runs)) * 100
    
    def get_average_duration(self, days: int = 7) -> float:
        """
        Calculate average pipeline duration.
        """
        runs = self.get_recent_runs(days)
        if not runs:
            return 0.0
        
        total_duration = sum(r['duration'] for r in runs)
        return total_duration / len(runs)
