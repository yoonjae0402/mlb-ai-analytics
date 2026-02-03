import pytest
from unittest.mock import patch, MagicMock
import os
import json
import datetime
from datetime import timedelta
import time
import requests

from src.data.fetcher import MLBDataFetcher, DataFetchError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Helper function to create dummy game data
def create_dummy_game_data(game_id: int, status_code: str, date_str: str = "2024-07-04") -> dict:
    return {
        "gamePk": game_id,
        "game_id": game_id,
        "game_date": date_str,
        "status": {"statusCode": status_code, "detailedState": "Final" if status_code == "F" else "Scheduled"},
        "home_team": {"name": "Home Team", "abbreviation": "HT"},
        "away_team": {"name": "Away Team", "abbreviation": "AT"},
        "home_score": 5 if status_code == "F" else 0,
        "away_score": 3 if status_code == "F" else 0,
        "venue": {"name": "Stadium"},
        "liveData": { # Simplified liveData for full detail
            "plays": {"allPlays": []},
            "boxscore": {"teams": []},
            "linescore": {"teams": {"home": {"runs": 5}, "away": {"runs": 3}}}
        },
        "gameData": { # Simplified gameData for full detail
            "datetime": {"officialDate": date_str},
            "status": {"statusCode": status_code, "detailedState": "Final" if status_code == "F" else "Scheduled"},
            "teams": {
                "home": {"name": "Home Team", "abbreviation": "HT", "record": {"wins": 10, "losses": 5}},
                "away": {"name": "Away Team", "abbreviation": "AT", "record": {"wins": 8, "losses": 7}},
            },
            "venue": {"name": "Stadium", "location": {"cityState": "Some City, ST"}}
        }
    }

@pytest.fixture
def fetcher_instance(tmp_path):
    """Fixture for MLBDataFetcher with a temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return MLBDataFetcher(cache_dir=str(cache_dir))

class TestMLBDataFetcher:

    @patch('statsapi.schedule')
    @patch('statsapi.get')
    def test_get_schedule_no_games(self, mock_statsapi_get, mock_statsapi_schedule, fetcher_instance):
        mock_statsapi_schedule.return_value = []
        games = fetcher_instance.get_schedule("2024-01-01")
        assert len(games) == 0
        mock_statsapi_schedule.assert_called_once()
        mock_statsapi_get.assert_not_called()

    @patch('statsapi.schedule')
    @patch('statsapi.get')
    def test_get_schedule_success(self, mock_statsapi_get, mock_statsapi_schedule, fetcher_instance):
        game_id_1 = 12345
        game_id_2 = 67890
        date_str = "2024-07-04"

        # Mock schedule to return two game summaries
        mock_statsapi_schedule.return_value = [
            {"game_id": game_id_1, "game_date": date_str, "status": {"statusCode": "S"}},
            {"game_id": game_id_2, "game_date": date_str, "status": {"statusCode": "S"}}
        ]
        
        # Mock get for game details
        mock_statsapi_get.side_effect = [
            create_dummy_game_data(game_id_1, "F", date_str),
            create_dummy_game_data(game_id_2, "F", date_str)
        ]

        games = fetcher_instance.get_schedule(date_str)

        assert len(games) == 2
        assert games[0]["game_id"] == game_id_1
        assert games[1]["game_id"] == game_id_2
        mock_statsapi_schedule.assert_called_once_with(start_date=date_str, end_date=date_str)

        # Check schedule caching
        schedule_cache_file = os.path.join(fetcher_instance.cache_dir, f"schedule_{date_str}_{date_str}.json")
        assert os.path.exists(schedule_cache_file)


    @patch('statsapi.linescore')
    @patch('statsapi.boxscore_data')
    def test_get_game_data_success(self, mock_boxscore, mock_linescore, fetcher_instance):
        game_id = 746969
        dummy_data = create_dummy_game_data(game_id, "F")
        mock_boxscore.return_value = dummy_data
        mock_linescore.return_value = "1-2-3 Final"

        details = fetcher_instance.get_game_data(game_id)
        assert details is not None
        assert details["game_id"] == game_id
        assert details["boxscore"] == dummy_data
        mock_boxscore.assert_called_once_with(game_id)
        mock_linescore.assert_called_once_with(game_id)

    @patch('statsapi.linescore')
    @patch('statsapi.boxscore_data')
    def test_get_game_data_cache_hit(self, mock_boxscore, mock_linescore, fetcher_instance):
        game_id = 746969
        date_str = "2024-07-04"
        cached_data = create_dummy_game_data(game_id, "F", date_str)
        # Manually save to cache using the proper cache structure
        cache_key = f"game_{game_id}"
        fetcher_instance._save_to_cache(cache_key, cached_data)

        details = fetcher_instance.get_game_data(game_id)
        # Should return cached data without API calls
        assert details == cached_data
        mock_boxscore.assert_not_called()
        mock_linescore.assert_not_called()

    @patch('statsapi.linescore')
    @patch('statsapi.boxscore_data')
    def test_get_game_data_cache_stale_non_final(self, mock_boxscore, mock_linescore, fetcher_instance):
        game_id = 746969
        date_str = "2024-07-04"
        
        # Create a stale cached non-final game by manually creating an old cache file
        cached_data = create_dummy_game_data(game_id, "I", date_str)
        old_timestamp = (datetime.datetime.now() - timedelta(minutes=10)).isoformat()

        cache_file = os.path.join(fetcher_instance.cache_dir, f"game_{game_id}.json")
        with open(cache_file, 'w') as f:
            json.dump({"timestamp": old_timestamp, "payload": cached_data}, f)

        # API call should happen because cache is stale
        fresh_data = create_dummy_game_data(game_id, "F", date_str)
        mock_boxscore.return_value = fresh_data
        mock_linescore.return_value = "Final Score"

        details = fetcher_instance.get_game_data(game_id)
        assert details["game_id"] == game_id
        mock_boxscore.assert_called_once() # API should be called

    @patch('statsapi.schedule')
    @patch('statsapi.get')
    def test_get_schedule_schedule_cache_stale(self, mock_statsapi_get, mock_statsapi_schedule, fetcher_instance):
        date_str = "2024-07-04"
        game_id_1 = 111
        game_id_2 = 222

        # Create a stale schedule cache with an in-progress game
        stale_schedule = [
            {"game_id": game_id_1, "game_date": date_str, "status": {"statusCode": "I"}},
            {"game_id": game_id_2, "game_date": date_str, "status": {"statusCode": "S"}}
        ]
        # Manually create stale cache with old timestamp
        cache_key = f"schedule_{date_str}_{date_str}"
        cache_file = os.path.join(fetcher_instance.cache_dir, f"{cache_key}.json")
        old_timestamp = (datetime.datetime.now() - timedelta(hours=2)).isoformat()
        with open(cache_file, 'w') as f:
            json.dump({"timestamp": old_timestamp, "payload": stale_schedule}, f)

        # Mock API calls when cache is considered stale
        mock_statsapi_schedule.return_value = [
            {"game_id": game_id_1, "game_date": date_str, "status": {"statusCode": "F"}},
            {"game_id": game_id_2, "game_date": date_str, "status": {"statusCode": "F"}}
        ]
        mock_statsapi_get.side_effect = [
            create_dummy_game_data(game_id_1, "F", date_str),
            create_dummy_game_data(game_id_2, "F", date_str)
        ]

        games = fetcher_instance.get_schedule(date_str)
        assert len(games) == 2
        mock_statsapi_schedule.assert_called_once()

        # Verify old cache is removed (or overwritten)
        schedule_cache_file = os.path.join(fetcher_instance.cache_dir, f"schedule_{date_str}_{date_str}.json")
        with open(schedule_cache_file, 'r') as f:
            updated_cache = json.load(f)
            assert updated_cache.get("timestamp") is not None
            assert (datetime.datetime.now() - datetime.datetime.fromisoformat(updated_cache["timestamp"])) < timedelta(minutes=1)

    @patch('statsapi.schedule')
    @patch('statsapi.get')
    def test_get_schedule_schedule_cache_valid_final_games(self, mock_statsapi_get, mock_statsapi_schedule, fetcher_instance):
        date_str = "2024-07-04"
        game_id_1 = 333
        game_id_2 = 444

        # Create a valid schedule cache with final games
        valid_schedule_games = [
            create_dummy_game_data(game_id_1, "F", date_str),
            create_dummy_game_data(game_id_2, "F", date_str)
        ]
        # fetcher_instance._save_to_cache({"games": valid_schedule_games, "cache_timestamp": datetime.datetime.now().isoformat()}, date_str)

        # Manually save schedule to cache
        cache_key = f"schedule_{date_str}_{date_str}"
        fetcher_instance._save_to_cache(cache_key, valid_schedule_games)

        # Also cache individual game details as get_schedule would do
        fetcher_instance._save_to_cache(f"game_{game_id_1}", valid_schedule_games[0])
        fetcher_instance._save_to_cache(f"game_{game_id_2}", valid_schedule_games[1])


        games = fetcher_instance.get_schedule(date_str)
        assert len(games) == 2
        mock_statsapi_schedule.assert_not_called() # Should not call API if schedule cache is valid
        mock_statsapi_get.assert_not_called() # Should not call API if game details are also valid and final

