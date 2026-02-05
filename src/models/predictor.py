"""
MLB AI Analytics - PyTorch Models

Neural network models for predicting player and game performance.

Models:
    - PlayerLSTM: LSTM with attention for time-series prediction
    - PlayerPredictor: Simple feedforward network
    - BattingPredictor: Specialized for batting stats
    - PitchingPredictor: Specialized for pitching stats
"""

from pathlib import Path
from typing import Optional
import logging

import torch
import torch.nn as nn
import numpy as np


logger = logging.getLogger(__name__)


class PlayerLSTM(nn.Module):
    """
    LSTM model with attention for player performance prediction.

    Architecture:
        - Bidirectional LSTM layers
        - Multi-head self-attention
        - Fully connected output layers

    Args:
        input_size: Number of input features per timestep
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        output_size: Number of output predictions
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        self._initialize_weights()

        logger.info(
            f"PlayerLSTM initialized: {input_size} -> LSTM({hidden_size}x{num_layers}) -> {output_size}"
        )

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_length, input_size)

        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Take last timestep
        last_hidden = attn_out[:, -1, :]

        # Output layers
        output = self.fc(last_hidden)

        return output

    def predict(
        self,
        features: np.ndarray | torch.Tensor,
        return_numpy: bool = True
    ) -> np.ndarray | torch.Tensor:
        """Make predictions."""
        self.eval()

        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)

        if features.dim() == 2:
            features = features.unsqueeze(0)

        with torch.no_grad():
            predictions = self.forward(features)

        if return_numpy:
            return predictions.numpy()
        return predictions

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "output_size": self.output_size,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PlayerLSTM":
        """Load model from checkpoint."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]

        model = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_size=config["output_size"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(f"Model loaded from {path}")
        return model


class PlayerPredictor(nn.Module):
    """
    Simple feedforward network for player performance prediction.

    Architecture:
        - Input layer
        - Hidden layers with BatchNorm, ReLU, Dropout
        - Output layer
    """

    def __init__(
        self,
        input_features: int = 15,
        hidden_dim: int = 64,
        output_dim: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def predict(
        self,
        features: np.ndarray | torch.Tensor,
        return_numpy: bool = True
    ) -> np.ndarray | torch.Tensor:
        """Make predictions."""
        self.eval()

        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)

        if features.dim() == 1:
            features = features.unsqueeze(0)

        with torch.no_grad():
            predictions = self.forward(features)

        if return_numpy:
            return predictions.numpy()
        return predictions

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "input_features": self.input_features,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> "PlayerPredictor":
        """Load model from disk."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]

        model = cls(
            input_features=config["input_features"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model


class BattingPredictor(PlayerPredictor):
    """Specialized predictor for batting statistics."""

    FEATURE_NAMES = [
        "rolling_avg_10", "rolling_obp_10", "rolling_slg_10",
        "rolling_hr_10", "vs_pitcher_avg", "home_away",
        "day_night", "park_factor", "recent_form", "days_rest",
    ]

    OUTPUT_NAMES = ["hits", "home_runs", "rbi", "walks"]

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__(
            input_features=len(self.FEATURE_NAMES),
            hidden_dim=hidden_dim,
            output_dim=len(self.OUTPUT_NAMES),
            dropout=dropout,
        )


class PitchingPredictor(PlayerPredictor):
    """Specialized predictor for pitching statistics."""

    FEATURE_NAMES = [
        "rolling_era_5", "rolling_whip_5", "rolling_k9_5",
        "vs_team_era", "home_away", "day_night",
        "park_factor", "days_rest", "pitch_count_last", "season_workload",
    ]

    OUTPUT_NAMES = ["innings_pitched", "strikeouts", "walks", "earned_runs"]

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__(
            input_features=len(self.FEATURE_NAMES),
            hidden_dim=hidden_dim,
            output_dim=len(self.OUTPUT_NAMES),
            dropout=dropout,
        )


class GameTransformer(nn.Module):
    """
    Transformer model for game outcome prediction.

    Uses team embeddings and game context features to predict
    win probability.
    """

    def __init__(
        self,
        num_teams: int = 30,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.team_embedding = nn.Embedding(num_teams, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # [away_win, home_win]
        )

    def forward(self, team_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            team_ids: Team ID tensor (batch, 2) for [away, home]
            features: Game features tensor (batch, seq, d_model)

        Returns:
            Win probability tensor (batch, 2)
        """
        team_emb = self.team_embedding(team_ids)  # (batch, 2, d_model)
        x = team_emb + features
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)
