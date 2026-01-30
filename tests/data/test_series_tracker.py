import pytest
from unittest.mock import MagicMock, patch
from src.data.series_tracker import SeriesTracker

class TestSeriesTracker:

    @pytest.fixture
    def tracker(self):
        mock_fetcher = MagicMock()
        return SeriesTracker(fetcher=mock_fetcher)

    def test_get_video_type_middle(self, tracker):
        # Setup: Game today, same teams play tomorrow
        game = {
            "home_team": "Yankees",
            "away_team": "Red Sox",
            "date": "2023-10-01"
        }
        
        # Mock tomorrow's schedule
        tracker.fetcher.get_schedule.return_value = [
            {"home_team": "Yankees", "away_team": "Red Sox", "summary": "NYY vs BOS"}
        ]
        tracker._get_tomorrow_date = MagicMock(return_value="2023-10-02")
        
        video_type = tracker.get_video_type(game)
        assert video_type == "series_middle"

    def test_get_video_type_end(self, tracker):
        # Setup: Game today, different teams play tomorrow
        game = {
            "home_team": "Yankees",
            "away_team": "Red Sox",
            "date": "2023-10-01"
        }
        
        # Mock tomorrow's schedule (Yankees play Orioles next, Red Sox play Rays)
        tracker.fetcher.get_schedule.return_value = [
            {"home_team": "Orioles", "away_team": "Yankees", "summary": "BAL vs NYY"},
            {"home_team": "Rays", "away_team": "Red Sox", "summary": "TB vs BOS"}
        ]
        tracker._get_tomorrow_date = MagicMock(return_value="2023-10-02")
        
        video_type = tracker.get_video_type(game)
        assert video_type == "series_end"
