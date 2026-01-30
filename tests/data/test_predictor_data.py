import pytest
from unittest.mock import MagicMock
import numpy as np
from src.data.predictor_data import PredictionDataProcessor

class TestPredictionDataProcessor:

    @pytest.fixture
    def processor(self):
        mock_fetcher = MagicMock()
        return PredictionDataProcessor(fetcher=mock_fetcher)

    def test_get_player_sequence_cross_season(self, processor):
        # Mock logic:
        # Request date: 2024-04-05
        # Current season (2024) has 2 games
        # Previous season (2023) has many games
        # Sequence length = 5
        # Needed: 3 games from 2023
        
        processor.fetcher.get_player_id.return_value = 12345
        
        # Mock 2024 logs
        processor.fetcher.get_player_game_logs.side_effect = [
            # First call for 2024
            [
                {"date": "2024-04-01", "stat": {"hits": 1}},
                {"date": "2024-04-02", "stat": {"hits": 0}}
            ],
            # Second call for 2023
            [
                {"date": "2023-09-28", "stat": {"hits": 1}},
                {"date": "2023-09-29", "stat": {"hits": 2}},
                {"date": "2023-09-30", "stat": {"hits": 1}},
                {"date": "2023-10-01", "stat": {"hits": 0}}
            ]
        ]
        
        sequence = processor.get_player_sequence("Judge", "2024-04-05", sequence_length=5)
        
        assert sequence.shape == (5, 11)
        # Verify season break feature (last feature, index 10)
        # The break should be between the 2023 games and 2024 games
        # Sequence: [2023-09-29, 2023-09-30, 2023-10-01, 2024-04-01, 2024-04-02]
        # The last game of prev season is index 2 (2023-10-01)
        # So index 2 should have season_break = 1
        
        assert sequence[2, 10] == 1.0 # Season break
        assert sequence[3, 10] == 0.0 # New season start
        assert sequence[4, 10] == 0.0

    def test_games_to_features(self, processor):
        games = [
            {"date": "2023-10-01", "batting_average": 0.300, "season_break": 1}
        ]
        features = processor._games_to_features(games, target_length=2)
        
        # Should zero pad
        assert features.shape == (2, 11)
        # First row should be zeros
        assert np.all(features[0] == 0)
        # Second row should be data
        assert features[1, 0] == pytest.approx(0.300)
        assert features[1, 10] == 1.0
