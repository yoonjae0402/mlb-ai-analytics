import pytest
from unittest.mock import MagicMock
from src.analysis.analyzer import MLBStatsAnalyzer

class TestMLBStatsAnalyzer:

    @pytest.fixture
    def analyzer(self):
        mock_fetcher = MagicMock()
        return MLBStatsAnalyzer(fetcher=mock_fetcher)

    def test_extract_key_insights(self, analyzer):
        game_data = {
            "game_id": 1,
            "home_score": 10,
            "away_score": 2,
            "home_team": "Yankees",
            "away_team": "Red Sox"
        }
        
        insights = analyzer._extract_key_insights(game_data)
        assert len(insights) > 0
        assert "Dominant victory" in insights[0]

    def test_find_top_performances(self, analyzer):
        game_data = {
            "box_score": {
                "home_batters": [
                    {"name": "Judge", "hits": 3, "home_runs": 2, "rbi": 4, "runs": 2}
                ],
                "away_batters": []
            },
            "home_team": "Yankees",
            "away_team": "Red Sox"
        }
        
        performances = analyzer._find_top_performances(game_data)
        assert len(performances) == 2
        assert performances[0]["player"] == "Judge"
        assert performances[0]["type"] == "batting"
