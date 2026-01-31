import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
from src.data.predictor_data import PredictionDataProcessor

class MLBDataset(Dataset):
    """
    PyTorch Dataset for MLB Player Performance Prediction.

    Features are pre-computed during initialization to avoid API calls during training.

    Expected Data Format:
    - samples: List of tuples (player_id, date, target_class)
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, int]],
        processor: PredictionDataProcessor,
        sequence_length: int = 10,
        precompute: bool = True
    ):
        """
        Args:
            samples: List of (player_name/id, prediction_date, target_label)
            processor: Instance of PredictionDataProcessor to fetch features
            sequence_length: Number of games in history window
            precompute: If True, pre-fetch all features during initialization (recommended)
        """
        self.samples = samples
        self.processor = processor
        self.sequence_length = sequence_length

        # Pre-compute all features to avoid API calls in __getitem__
        self.features_cache = {}
        self.valid_indices = []

        if precompute:
            self._precompute_features()

    def _precompute_features(self):
        """
        Pre-fetch all player sequences to avoid blocking the training loop.
        This should be called once during dataset initialization.
        """
        print(f"Pre-computing features for {len(self.samples)} samples...")

        for idx, (player, date, target) in enumerate(self.samples):
            try:
                features = self.processor.get_player_sequence(
                    player,
                    date,
                    sequence_length=self.sequence_length
                )

                # Store in cache with sample index as key
                self.features_cache[idx] = {
                    'features': np.array(features, dtype=np.float32),
                    'target': target
                }
                self.valid_indices.append(idx)

                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(self.samples)} samples...")

            except Exception as e:
                # Skip samples that fail to load
                print(f"  Warning: Failed to load sample {idx} ({player}, {date}): {e}")
                continue

        print(f"Pre-computation complete. {len(self.valid_indices)}/{len(self.samples)} samples valid.")

    def __len__(self):
        # Return count of valid samples only
        if self.features_cache:
            return len(self.valid_indices)
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a pre-computed sample. No API calls happen here.
        """
        # Map to valid index if using cache
        if self.features_cache:
            if idx >= len(self.valid_indices):
                raise IndexError(f"Index {idx} out of range for {len(self.valid_indices)} valid samples")

            original_idx = self.valid_indices[idx]
            cached = self.features_cache[original_idx]

            x_tensor = torch.tensor(cached['features'], dtype=torch.float32)
            y_tensor = torch.tensor(cached['target'], dtype=torch.long)

            return x_tensor, y_tensor

        # Fallback for non-precomputed mode (not recommended for training)
        player, date, target = self.samples[idx]

        try:
            features = self.processor.get_player_sequence(
                player,
                date,
                sequence_length=self.sequence_length
            )

            x_tensor = torch.tensor(features, dtype=torch.float32)
            y_tensor = torch.tensor(target, dtype=torch.long)

            return x_tensor, y_tensor

        except Exception as e:
            # Handle error gracefully (return zero tensor)
            return torch.zeros((self.sequence_length, 11), dtype=torch.float32), torch.tensor(-1, dtype=torch.long)

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
