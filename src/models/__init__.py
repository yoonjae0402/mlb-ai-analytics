"""
MLB Video Pipeline - Models Package

PyTorch-based prediction models for player performance:
- classifier: LSTM classifier (3-class: below/average/above)
- dataset: PyTorch dataset for ML model
- trainer: Training loop and evaluation

Usage:
    from src.models import PlayerPerformanceLSTM, MLBDataset, ModelTrainer

    # Create classifier model
    model = PlayerPerformanceLSTM(input_size=11, hidden_size=64)

    # Train model
    trainer = ModelTrainer(model)
    trainer.train(train_loader, val_loader, epochs=50)

    # Make predictions
    predictions = model.predict(features)
"""

# Import from both old and new locations for compatibility
from src.models.classifier import PlayerPerformanceLSTM
from src.models.dataset import MLBDataset, transform_target
from src.models.trainer import ModelTrainer
from src.models.explainer import PredictionExplainer

__all__ = [
    "PlayerPerformanceLSTM",
    "MLBDataset",
    "transform_target",
    "ModelTrainer",
    "PredictionExplainer",
]
