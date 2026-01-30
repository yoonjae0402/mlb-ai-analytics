import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset
from src.models.trainer import ModelTrainer
from src.models.explainer import PredictionExplainer
from src.models.classifier import PlayerPerformanceLSTM

class TestTrainer:
    
    @pytest.fixture
    def model_setup(self):
        model = PlayerPerformanceLSTM(input_size=11, hidden_size=16, num_layers=1)
        # Create dummy dataset
        x = torch.randn(20, 10, 11)
        y = torch.randint(0, 3, (20,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)
        return model, loader

    def test_train_epoch_runs(self, model_setup, tmp_path):
        model, loader = model_setup
        trainer = ModelTrainer(model, checkpoint_dir=str(tmp_path))
        
        history = trainer.train(loader, loader, epochs=1)
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == 1
        assert (tmp_path / "best_model.pt").exists()

    def test_evaluate(self, model_setup):
        model, loader = model_setup
        trainer = ModelTrainer(model)
        
        metrics = trainer.evaluate(loader)
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0

class TestExplainer:
    
    def test_explain_above_average(self):
        explainer = PredictionExplainer()
        # High stats: BA=.350, OBP=.400, Home=1
        seq = np.zeros((10, 11))
        seq[:, 0] = 0.350
        seq[:, 1] = 0.400
        seq[:, 9] = 1.0
        
        explanation = explainer.explain_prediction(seq, prediction_class=2, confidence=0.85)
        
        assert explanation["prediction"] == "Above Average"
        assert "confidence" in explanation
        assert len(explanation["reasons"]) > 0
        assert any("Batting Average" in r for r in explanation["reasons"])
        assert any("Home field" in r for r in explanation["reasons"])

    def test_explain_below_average(self):
        explainer = PredictionExplainer()
        # Low stats: BA=.150, High Strikeouts
        seq = np.zeros((10, 11))
        seq[:, 0] = 0.150
        seq[:, 8] = 2.0 
        
        explanation = explainer.explain_prediction(seq, prediction_class=0, confidence=0.85)
        
        assert explanation["prediction"] == "Below Average"
        assert any("Cold streak" in r for r in explanation["reasons"])
