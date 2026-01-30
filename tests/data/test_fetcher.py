import pytest
import os
from unittest.mock import patch, MagicMock
from src.data.fetcher import MLBDataFetcher

class TestMLBDataFetcher:
    
    @pytest.fixture
    def fetcher(self, tmp_path):
        # Use a temporary directory for cache to avoid messing with real data
        return MLBDataFetcher(cache_dir=str(tmp_path / "cache"))

    @patch("statsapi.schedule")
    def test_get_schedule(self, mock_schedule, fetcher):
        # Mock response
        mock_schedule.return_value = [{"game_id": 12345, "summary": "NYY vs BOS"}]
        
        games = fetcher.get_schedule("10/10/2023")
        
        assert len(games) == 1
        assert games[0]["game_id"] == 12345
        mock_schedule.assert_called_once_with(start_date="10/10/2023", end_date="10/10/2023")

    @patch("statsapi.boxscore_data")
    @patch("statsapi.linescore")
    def test_get_game_data(self, mock_linescore, mock_boxscore, fetcher):
        mock_boxscore.return_value = {"teams": {"home": {}, "away": {}}}
        mock_linescore.return_value = "Run 1"
        
        data = fetcher.get_game_data(12345)
        
        assert data["game_id"] == 12345
        assert "boxscore" in data
        assert "linescore" in data

    def test_caching(self, fetcher):
        # Manually save something to cache
        cache_key = "test_key"
        fetcher._save_to_cache(cache_key, {"value": 1})
        
        # Manually load it back
        loaded = fetcher._load_from_cache(cache_key, "schedule")
        assert loaded == {"value": 1}

