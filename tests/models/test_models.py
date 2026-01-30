import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from src.models.classifier import PlayerPerformanceLSTM
from src.models.dataset import MLBDataset, transform_target

class TestClassifier:
    
    def test_model_output_shape(self):
        batch_size = 4
        seq_len = 10
        features = 11
        classes = 3
        
        model = PlayerPerformanceLSTM(input_size=features, num_classes=classes)
        x = torch.randn(batch_size, seq_len, features)
        
        logits = model(x)
        assert logits.shape == (batch_size, classes)

    def test_model_prediction(self):
        model = PlayerPerformanceLSTM()
        x = torch.randn(1, 10, 11)
        probs = model.predict(x)
        assert probs.shape == (1, 3)
        assert torch.isclose(probs.sum(), torch.tensor(1.0))

class TestDataset:
    
    def test_dataset_getitem(self):
        mock_processor = MagicMock()
        # Mock features: 10 games, 11 features
        mock_processor.get_player_sequence.return_value = np.zeros((10, 11))
        
        samples = [("Judge", "2024-05-01", 2)]
        dataset = MLBDataset(samples, mock_processor)
        
        x, y = dataset[0]
        
        assert x.shape == (10, 11)
        assert y == 2
        mock_processor.get_player_sequence.assert_called_with("Judge", "2024-05-01", sequence_length=10)

    def test_target_transform(self):
        assert transform_target({'hits': 1, 'home_runs': 0, 'rbi': 0}) == 1 # Average
        assert transform_target({'hits': 0, 'home_runs': 0, 'rbi': 0}) == 0 # Below
        assert transform_target({'hits': 1, 'home_runs': 1, 'rbi': 0}) == 2 # Above (score 4)
