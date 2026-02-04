
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PipelineMonitor:
    """
    Tracks pipeline performance, costs, and success rates.
    Persists metrics to JSON for monitoring dashboards.
    """
    
    METRICS_FILE = Path("logs/pipeline_metrics.json")
    
    def __init__(self):
        self.METRICS_FILE.parent.mkdir(exist_ok=True)
        
    def log_run(
        self,
        date: str,
        team: str,
        mode: str,
        success: bool,
        duration_seconds: float,
        costs: Dict[str, float],
        error: Optional[str] = None
    ):
        """
        Log a pipeline execution run.
        """
        run_data = {
            "timestamp": datetime.now().isoformat(),
            "date_processed": date,
            "team": team,
            "mode": mode,
            "success": success,
            "duration": round(duration_seconds, 2),
            "costs": costs,
            "total_cost": sum(costs.values()),
            "error": error
        }
        
        # Log to file (append)
        try:
            self._append_to_json(run_data)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            
        # Log to console
        status = "SUCCESS" if success else "FAILURE"
        logger.info(f"Pipeline Run [{status}] - Duration: {duration_seconds:.2f}s - Cost: ${run_data['total_cost']:.4f}")
        
    def _append_to_json(self, data: Dict):
        """Append record to JSON line file."""
        with open(self.METRICS_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")

    def get_summary(self, last_n: int = 10) -> Dict:
        """Get success rate and avg cost of last N runs."""
        runs = []
        if not self.METRICS_FILE.exists():
            return {}
            
        try:
            with open(self.METRICS_FILE, "r") as f:
                 for line in f:
                     if line.strip():
                        runs.append(json.loads(line))
        except Exception:
            return {}
            
        recent = runs[-last_n:]
        if not recent:
            return {}
            
        success_count = sum(1 for r in recent if r['success'])
        avg_cost = sum(r['total_cost'] for r in recent) / len(recent)
        
        return {
            "runs": len(recent),
            "success_rate": success_count / len(recent),
            "avg_cost": avg_cost
        }
