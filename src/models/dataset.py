import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
from src.data.predictor_data import PredictionDataProcessor

class MLBDataset(Dataset):
    """
    PyTorch Dataset for MLB Player Performance Prediction.
    
    Expected Data Format:
    - samples: List of tuples (player_id, date, target_class)
    """
    
    def __init__(
        self, 
        samples: List[Tuple[str, str, int]], 
        processor: PredictionDataProcessor,
        sequence_length: int = 10
    ):
        """
        Args:
            samples: List of (player_name/id, prediction_date, target_label)
            processor: Instance of PredictionDataProcessor to fetch features
            sequence_length: Number of games in history window
        """
        self.samples = samples
        self.processor = processor
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        player, date, target = self.samples[idx]
        
        # Get feature sequence (seq_len, 11)
        # This might do API calls, which is slow for a Dataset __getitem__
        # In a real training loop, we'd pre-process this offline.
        # For Phase 2 prototype, we will assume we can cache or this is valid for MVP.
        # Ideally, 'samples' should eventually store the pre-computed arrays.
        
        try:
            features = self.processor.get_player_sequence(
                player, 
                date, 
                sequence_length=self.sequence_length
            )
            
            # Convert to float tensor
            x_tensor = torch.tensor(features, dtype=torch.float32)
            y_tensor = torch.tensor(target, dtype=torch.long)
            
            return x_tensor, y_tensor
            
        except Exception as e:
            # Handle error gracefully (return zero tensor)
            # In production training, we should filter these out beforehand
            return torch.zeros((self.sequence_length, 11)), torch.tensor(-1)

def transform_target(stats: Dict) -> int:
    """
    Heuristic to define target label from next game stats.
    0: Below Average
    1: Average
    2: Above Average
    """
    # Simple heuristic based on fantasy points or key stats
    hits = stats.get('hits', 0)
    hr = stats.get('home_runs', 0)
    rbi = stats.get('rbi', 0)
    
    score = hits + (hr * 3) + rbi
    
    if score >= 3:
        return 2 # Above Average (e.g. HR or 3 hits or 1 hit + 2 RBI)
    elif score >= 1:
        return 1 # Average (e.g. 1 hit)
    else:
        return 0 # Below Average
