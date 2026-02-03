import pytest
from pathlib import Path
from src.utils import MetricsCollector, AlertManager

class TestMetricsCollector:
    
    @pytest.fixture
    def metrics(self, tmp_path):
        return MetricsCollector(metrics_file=str(tmp_path / "metrics.jsonl"))
    
    def test_log_pipeline_run(self, metrics):
        """Test logging a pipeline run."""
        metrics.log_pipeline_run(
            date="2024-07-04",
            team="Yankees",
            success=True,
            duration=120.5,
            costs={"gemini": 0, "audio": 0, "total": 0}
        )
        
        runs = metrics.get_recent_runs(days=7)
        assert len(runs) == 1
        assert runs[0]["team"] == "Yankees"
        assert runs[0]["success"] is True
    
    def test_get_cost_summary(self, metrics):
        """Test cost summary calculation."""
        # Log multiple runs
        metrics.log_pipeline_run(
            date="2024-07-04",
            team="Yankees",
            success=True,
            duration=120.0,
            costs={"gemini": 0, "audio": 0, "total": 0}
        )
        metrics.log_pipeline_run(
            date="2024-07-05",
            team="Red Sox",
            success=True,
            duration=115.0,
            costs={"gemini": 0, "audio": 0, "total": 0}
        )
        
        from datetime import datetime, timedelta
        start = (datetime.now() - timedelta(days=1)).isoformat()
        end = datetime.now().isoformat()
        
        summary = metrics.get_cost_summary(start, end)
        assert summary["runs"] == 2
        assert summary["total"] == 0  # All free!
    
    def test_success_rate(self, metrics):
        """Test success rate calculation."""
        metrics.log_pipeline_run(
            date="2024-07-04",
            team="Yankees",
            success=True,
            duration=120.0,
            costs={"total": 0}
        )
        metrics.log_pipeline_run(
            date="2024-07-05",
            team="Red Sox",
            success=False,
            duration=60.0,
            costs={"total": 0},
            error="Test error"
        )
        
        rate = metrics.get_success_rate(days=7)
        assert rate == 50.0

class TestAlertManager:
    
    def test_check_cost_threshold_below(self):
        """Test cost threshold when below limit."""
        alerts = AlertManager()
        result = alerts.check_cost_threshold(0.50, threshold=1.00)
        assert result is False
    
    def test_check_cost_threshold_above(self):
        """Test cost threshold when above limit."""
        alerts = AlertManager()
        result = alerts.check_cost_threshold(1.50, threshold=1.00)
        assert result is True
