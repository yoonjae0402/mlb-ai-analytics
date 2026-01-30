import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PlayerPerformanceLSTM(nn.Module):
    """
    LSTM-based classifier for player performance prediction.
    
    Architecture:
    - LSTM Layer: Processes the sequence of game features
    - Dense Layers: Maps LSTM output to class logits
    
    Inputs:
    - x: (batch_size, sequence_length, input_size)
    
    Outputs:
    - logits: (batch_size, num_classes)
    """
    
    def __init__(
        self, 
        input_size: int = 11, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super(PlayerPerformanceLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Batch normalization for input features
        self.bn_input = nn.BatchNorm1d(input_size)
        
        # LSTM Model
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification Head
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
           x: Input tensor of shape (batch, seq_len, features)
           
        Returns:
           Logits of shape (batch, num_classes)
        """
        # x shape: (batch, seq, feature)
        batch_size, seq_len, features = x.shape
        
        # Batch Norm expects (batch, features, seq_len) for 1D or (batch, features) 
        # But we have time series. Often better to just BN the input features dimension.
        # Reshape to (batch * seq, features) for BN, then back
        x_reshaped = x.view(-1, features)
        x_bn = self.bn_input(x_reshaped)
        x = x_bn.view(batch_size, seq_len, features)
        
        # LSTM forward
        # out: (batch, seq, hidden_size)
        # hn: (num_layers, batch, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We take the output of the last time step
        # shape: (batch, hidden_size)
        last_step_out = lstm_out[:, -1, :]
        
        # Classification head
        logits = self.fc_layers(last_step_out)
        
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for inference (returns probabilities)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for saving."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes
        }
