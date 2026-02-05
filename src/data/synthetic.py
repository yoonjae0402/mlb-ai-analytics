"""
Synthetic data generation for MLB player performance sequences.

Generates realistic batting data with hot/cold streaks,
correlated targets, and player archetypes.
"""

import numpy as np
import pandas as pd
from typing import Optional


# Feature names matching model input
FEATURE_NAMES = [
    "batting_avg", "on_base_pct", "slugging_pct", "wOBA",
    "barrel_rate", "exit_velocity", "launch_angle", "sprint_speed",
    "k_rate", "bb_rate", "hard_hit_rate", "pull_rate",
    "park_factor", "platoon_advantage", "days_rest",
]

TARGET_NAMES = ["hits", "home_runs", "rbi", "walks"]

# Archetypes define base stat distributions
PLAYER_ARCHETYPES = {
    "power": {
        "base_mean": [0.245, 0.330, 0.500, 0.360, 0.12, 91.0, 18.0, 27.5, 0.28, 0.10, 0.42, 0.45, 1.0, 0.5, 1.0],
        "target_mean": [1.0, 0.18, 0.85, 0.45],
    },
    "contact": {
        "base_mean": [0.300, 0.365, 0.420, 0.355, 0.06, 87.0, 12.0, 28.0, 0.14, 0.08, 0.35, 0.38, 1.0, 0.5, 1.0],
        "target_mean": [1.5, 0.05, 0.55, 0.35],
    },
    "balanced": {
        "base_mean": [0.270, 0.345, 0.450, 0.350, 0.08, 89.0, 14.0, 27.8, 0.20, 0.09, 0.38, 0.40, 1.0, 0.5, 1.0],
        "target_mean": [1.2, 0.10, 0.70, 0.40],
    },
    "speedster": {
        "base_mean": [0.275, 0.340, 0.390, 0.330, 0.04, 86.0, 10.0, 29.5, 0.18, 0.07, 0.30, 0.35, 1.0, 0.5, 1.0],
        "target_mean": [1.3, 0.03, 0.45, 0.30],
    },
}

# Preset players for the Prediction Explorer
PRESET_PLAYERS = {
    "Aaron Judge": {"archetype": "power", "hot_streak_prob": 0.35, "cold_streak_prob": 0.20},
    "Shohei Ohtani": {"archetype": "power", "hot_streak_prob": 0.40, "cold_streak_prob": 0.15},
    "Mookie Betts": {"archetype": "balanced", "hot_streak_prob": 0.30, "cold_streak_prob": 0.20},
    "Luis Arraez": {"archetype": "contact", "hot_streak_prob": 0.30, "cold_streak_prob": 0.15},
    "Ronald Acuna Jr.": {"archetype": "speedster", "hot_streak_prob": 0.35, "cold_streak_prob": 0.20},
    "Freddie Freeman": {"archetype": "balanced", "hot_streak_prob": 0.30, "cold_streak_prob": 0.18},
    "Juan Soto": {"archetype": "power", "hot_streak_prob": 0.30, "cold_streak_prob": 0.18},
    "Corey Seager": {"archetype": "power", "hot_streak_prob": 0.28, "cold_streak_prob": 0.22},
}


def generate_player_sequences(
    n_players: int = 200,
    seq_length: int = 10,
    n_features: int = 15,
    n_targets: int = 4,
    seed: Optional[int] = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic player performance sequences.

    Returns:
        X: (n_players, seq_length, n_features) feature sequences
        y: (n_players, n_targets) target values
    """
    rng = np.random.RandomState(seed)
    archetypes = list(PLAYER_ARCHETYPES.values())

    X = np.zeros((n_players, seq_length, n_features))
    y = np.zeros((n_players, n_targets))

    for i in range(n_players):
        arch = archetypes[rng.randint(len(archetypes))]
        base = np.array(arch["base_mean"])
        target_base = np.array(arch["target_mean"])

        # Determine streak state
        streak_roll = rng.random()
        if streak_roll < 0.25:
            streak_mult = 1.15  # hot
        elif streak_roll < 0.45:
            streak_mult = 0.85  # cold
        else:
            streak_mult = 1.0   # normal

        # Generate sequence with temporal correlation
        noise_scale = np.array([
            0.030, 0.025, 0.050, 0.030, 0.03, 2.0, 3.0, 0.5,
            0.04, 0.03, 0.05, 0.05, 0.15, 1.0, 2.0,
        ])

        for t in range(seq_length):
            # Gradual trend within sequence
            trend = 1.0 + (t - seq_length / 2) * 0.005 * (streak_mult - 1.0) * 3
            noise = rng.randn(n_features) * noise_scale
            X[i, t, :] = base * trend * streak_mult + noise

        # Targets correlated with recent performance
        recent = X[i, -3:, :].mean(axis=0)
        perf_factor = (recent[:4].mean() / base[:4].mean()) if base[:4].mean() > 0 else 1.0
        y[i, :] = target_base * perf_factor * streak_mult + rng.randn(n_targets) * 0.1

    # Clamp targets to reasonable ranges
    y[:, 0] = np.clip(y[:, 0], 0, 5)    # hits
    y[:, 1] = np.clip(y[:, 1], 0, 3)    # home_runs
    y[:, 2] = np.clip(y[:, 2], 0, 6)    # rbi
    y[:, 3] = np.clip(y[:, 3], 0, 4)    # walks

    return X.astype(np.float32), y.astype(np.float32)


def generate_player_sequence(
    archetype: str = "balanced",
    hot_streak_prob: float = 0.3,
    cold_streak_prob: float = 0.2,
    seq_length: int = 10,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a single player's performance sequence.

    Returns:
        X: (1, seq_length, n_features) feature sequence
        y: (1, n_targets) target values
    """
    rng = np.random.RandomState(seed)
    arch = PLAYER_ARCHETYPES.get(archetype, PLAYER_ARCHETYPES["balanced"])
    base = np.array(arch["base_mean"])
    target_base = np.array(arch["target_mean"])

    streak_roll = rng.random()
    if streak_roll < hot_streak_prob:
        streak_mult = 1.15
    elif streak_roll < hot_streak_prob + cold_streak_prob:
        streak_mult = 0.85
    else:
        streak_mult = 1.0

    noise_scale = np.array([
        0.030, 0.025, 0.050, 0.030, 0.03, 2.0, 3.0, 0.5,
        0.04, 0.03, 0.05, 0.05, 0.15, 1.0, 2.0,
    ])

    X = np.zeros((1, seq_length, len(base)))
    for t in range(seq_length):
        trend = 1.0 + (t - seq_length / 2) * 0.005 * (streak_mult - 1.0) * 3
        noise = rng.randn(len(base)) * noise_scale
        X[0, t, :] = base * trend * streak_mult + noise

    recent = X[0, -3:, :].mean(axis=0)
    perf_factor = (recent[:4].mean() / base[:4].mean()) if base[:4].mean() > 0 else 1.0
    y = (target_base * perf_factor * streak_mult + rng.randn(len(target_base)) * 0.1).reshape(1, -1)

    y[0, 0] = np.clip(y[0, 0], 0, 5)
    y[0, 1] = np.clip(y[0, 1], 0, 3)
    y[0, 2] = np.clip(y[0, 2], 0, 6)
    y[0, 3] = np.clip(y[0, 3], 0, 4)

    return X.astype(np.float32), y.astype(np.float32)


def generate_live_game_data(n_games: int = 8, seed: Optional[int] = None) -> list[dict]:
    """
    Generate synthetic live game data for the real-time dashboard demo.

    Returns:
        List of game dicts with teams, scores, inning, win probabilities.
    """
    rng = np.random.RandomState(seed)

    teams = [
        ("NYY", "New York Yankees"), ("BOS", "Boston Red Sox"),
        ("LAD", "Los Angeles Dodgers"), ("SF", "San Francisco Giants"),
        ("HOU", "Houston Astros"), ("ATL", "Atlanta Braves"),
        ("NYM", "New York Mets"), ("PHI", "Philadelphia Phillies"),
        ("SD", "San Diego Padres"), ("CHC", "Chicago Cubs"),
        ("SEA", "Seattle Mariners"), ("TB", "Tampa Bay Rays"),
        ("MIN", "Minnesota Twins"), ("TEX", "Texas Rangers"),
        ("BAL", "Baltimore Orioles"), ("CLE", "Cleveland Guardians"),
    ]

    rng.shuffle(teams)
    games = []

    for g in range(min(n_games, len(teams) // 2)):
        away = teams[g * 2]
        home = teams[g * 2 + 1]
        inning = rng.randint(1, 10)
        half = rng.choice(["Top", "Bottom"])
        away_score = int(rng.poisson(3.5))
        home_score = int(rng.poisson(3.8))

        # Win probability based on score diff and inning
        score_diff = home_score - away_score
        leverage = inning / 9.0
        home_wp = 1.0 / (1.0 + np.exp(-0.5 * score_diff * leverage))
        home_wp = float(np.clip(home_wp + rng.randn() * 0.05, 0.05, 0.95))

        # Generate inning-by-inning WP history
        wp_history = [0.5]
        for inn in range(1, inning + 1):
            delta = (home_wp - 0.5) / inning + rng.randn() * 0.08
            wp_history.append(float(np.clip(wp_history[-1] + delta, 0.05, 0.95)))
        wp_history[-1] = home_wp

        games.append({
            "game_id": f"demo_{g}",
            "away_abbrev": away[0],
            "away_name": away[1],
            "home_abbrev": home[0],
            "home_name": home[1],
            "away_score": away_score,
            "home_score": home_score,
            "inning": inning,
            "half": half,
            "status": "In Progress" if inning < 9 else "Final",
            "home_win_prob": home_wp,
            "wp_history": wp_history,
        })

    return games
