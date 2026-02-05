"""
Feature engineering: convert raw MLB API game logs into model input tensors.

Takes game log data (from statsapi or synthetic) and produces
(1, seq_length, n_features) tensors suitable for PlayerLSTM inference.
"""

import numpy as np
from typing import Optional

from src.data.synthetic import FEATURE_NAMES


def game_logs_to_tensor(
    game_logs: list[dict],
    seq_length: int = 10,
) -> np.ndarray:
    """
    Convert a list of game log dicts into a model input tensor.

    Each game log dict should have keys matching FEATURE_NAMES where available.
    Missing features are filled with league-average defaults.

    Args:
        game_logs: List of dicts, most recent last.
        seq_length: Number of timesteps the model expects.

    Returns:
        (1, seq_length, n_features) float32 array.
    """
    defaults = _league_average_defaults()
    n_features = len(FEATURE_NAMES)

    rows = []
    for log in game_logs[-seq_length:]:
        row = np.array([
            float(log.get(feat, defaults.get(feat, 0.0)))
            for feat in FEATURE_NAMES
        ], dtype=np.float32)
        rows.append(row)

    # Pad if fewer games than seq_length
    while len(rows) < seq_length:
        rows.insert(0, np.array([defaults.get(f, 0.0) for f in FEATURE_NAMES], dtype=np.float32))

    X = np.array(rows, dtype=np.float32).reshape(1, seq_length, n_features)
    return X


def stats_to_game_log(
    batting_avg: float = 0.265,
    on_base_pct: float = 0.330,
    slugging_pct: float = 0.430,
    woba: Optional[float] = None,
    barrel_rate: float = 0.07,
    exit_velocity: float = 88.5,
    launch_angle: float = 12.5,
    sprint_speed: float = 27.5,
    k_rate: float = 0.22,
    bb_rate: float = 0.08,
    hard_hit_rate: float = 0.35,
    pull_rate: float = 0.40,
    park_factor: float = 1.0,
    platoon_advantage: float = 0.0,
    days_rest: float = 1.0,
) -> dict:
    """Build a single game log dict from explicit stats."""
    if woba is None:
        woba = 0.7 * on_base_pct + 0.3 * slugging_pct

    return {
        "batting_avg": batting_avg,
        "on_base_pct": on_base_pct,
        "slugging_pct": slugging_pct,
        "wOBA": woba,
        "barrel_rate": barrel_rate,
        "exit_velocity": exit_velocity,
        "launch_angle": launch_angle,
        "sprint_speed": sprint_speed,
        "k_rate": k_rate,
        "bb_rate": bb_rate,
        "hard_hit_rate": hard_hit_rate,
        "pull_rate": pull_rate,
        "park_factor": park_factor,
        "platoon_advantage": platoon_advantage,
        "days_rest": days_rest,
    }


def _league_average_defaults() -> dict:
    """League-average defaults for each feature."""
    return {
        "batting_avg": 0.248,
        "on_base_pct": 0.317,
        "slugging_pct": 0.407,
        "wOBA": 0.315,
        "barrel_rate": 0.068,
        "exit_velocity": 88.4,
        "launch_angle": 12.1,
        "sprint_speed": 27.1,
        "k_rate": 0.224,
        "bb_rate": 0.083,
        "hard_hit_rate": 0.346,
        "pull_rate": 0.397,
        "park_factor": 1.0,
        "platoon_advantage": 0.0,
        "days_rest": 1.0,
    }
