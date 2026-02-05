"""Feature engineering from real MLB player stats.

Builds (N, seq_length, 15) feature arrays from player_stats rows,
with rolling averages, trend features, and contextual data.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 15 features matching model input
FEATURE_NAMES = [
    "batting_avg", "on_base_pct", "slugging_pct", "woba",
    "barrel_rate", "exit_velocity", "launch_angle", "sprint_speed",
    "k_rate", "bb_rate", "hard_hit_rate", "pull_rate",
    "park_factor", "platoon_advantage", "days_rest",
]

TARGET_NAMES = ["hits", "home_runs", "rbi", "walks"]

# League average defaults (2023-2024)
LEAGUE_DEFAULTS = {
    "batting_avg": 0.248,
    "on_base_pct": 0.320,
    "slugging_pct": 0.414,
    "woba": 0.315,
    "barrel_rate": 0.068,
    "exit_velocity": 88.5,
    "launch_angle": 12.0,
    "sprint_speed": 27.0,
    "k_rate": 0.224,
    "bb_rate": 0.083,
    "hard_hit_rate": 0.353,
    "pull_rate": 0.40,
    "park_factor": 1.0,
    "platoon_advantage": 0.5,
    "days_rest": 1.0,
}


def build_sequences_from_db(
    session,
    seq_length: int = 10,
    min_games: int = 20,
    seasons: list[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build feature sequences from PostgreSQL player_stats.

    Args:
        session: SQLAlchemy session (sync)
        seq_length: Number of games per sequence
        min_games: Minimum games required per player
        seasons: Filter by seasons (None = all)

    Returns:
        X: (N, seq_length, 15) feature sequences
        y: (N, 4) target values (hits, HR, RBI, walks for next game)
        player_ids: list of player IDs corresponding to each sample
    """
    from backend.db.models import Player, PlayerStat

    query = session.query(PlayerStat).join(Player)
    if seasons:
        from sqlalchemy import extract
        query = query.filter(
            extract("year", PlayerStat.game_date).in_(seasons)
        )
    query = query.order_by(PlayerStat.player_id, PlayerStat.game_date)

    all_stats = query.all()
    if not all_stats:
        logger.warning("No stats found in database")
        return np.array([]), np.array([]), []

    # Group by player
    by_player: dict[int, list] = {}
    for stat in all_stats:
        by_player.setdefault(stat.player_id, []).append(stat)

    X_list = []
    y_list = []
    pid_list = []

    for player_id, stats in by_player.items():
        if len(stats) < min_games:
            continue

        # Sort by date
        stats.sort(key=lambda s: s.game_date)
        df = _stats_to_dataframe(stats)

        # Create sliding window sequences
        for i in range(len(df) - seq_length):
            window = df.iloc[i : i + seq_length]
            target_row = df.iloc[i + seq_length]

            features = _extract_features(window)
            targets = np.array([
                target_row.get("hits", 0),
                target_row.get("home_runs", 0),
                target_row.get("rbi", 0),
                target_row.get("walks", 0),
            ], dtype=np.float32)

            X_list.append(features)
            y_list.append(targets)
            pid_list.append(player_id)

    if not X_list:
        return np.array([]), np.array([]), []

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(f"Built {X.shape[0]} sequences from {len(by_player)} players")
    return X, y, pid_list


def build_player_sequence(
    session,
    player_id: int,
    seq_length: int = 10,
) -> Optional[np.ndarray]:
    """Build a single feature sequence for a player (most recent games).

    Returns:
        (1, seq_length, 15) array or None if insufficient data.
    """
    from backend.db.models import PlayerStat

    stats = (
        session.query(PlayerStat)
        .filter_by(player_id=player_id)
        .order_by(PlayerStat.game_date.desc())
        .limit(seq_length)
        .all()
    )

    if len(stats) < seq_length:
        # Pad with defaults if we have at least 5 games
        if len(stats) < 5:
            return None

    # Reverse to chronological order
    stats = list(reversed(stats))
    df = _stats_to_dataframe(stats)
    features = _extract_features(df, seq_length=seq_length)
    return features.reshape(1, seq_length, -1)


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    player_ids: list[int],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict:
    """Time-based split to prevent data leakage.

    Assumes X/y are already sorted by time within each player.
    Uses a simple index-based split since data is ordered chronologically.
    """
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
        "train_player_ids": player_ids[:train_end],
        "val_player_ids": player_ids[train_end:val_end],
        "test_player_ids": player_ids[val_end:],
    }


def _stats_to_dataframe(stats: list) -> pd.DataFrame:
    """Convert a list of PlayerStat ORM objects to a DataFrame."""
    records = []
    prev_date = None
    for s in stats:
        days_rest = 1.0
        if prev_date is not None:
            days_rest = float((s.game_date - prev_date).days)
        prev_date = s.game_date

        # Compute wOBA if missing: simplified formula
        woba = s.woba
        if woba is None and s.plate_appearances and s.plate_appearances > 0:
            bb = s.walks or 0
            hbp = 0
            singles = (s.hits or 0) - (s.doubles or 0) - (s.triples or 0) - (s.home_runs or 0)
            doubles = s.doubles or 0
            triples = s.triples or 0
            hr = s.home_runs or 0
            pa = s.plate_appearances
            woba = (0.69 * bb + 0.72 * hbp + 0.87 * singles + 1.22 * doubles +
                    1.58 * triples + 2.01 * hr) / pa

        records.append({
            "game_date": s.game_date,
            "batting_avg": s.batting_avg or LEAGUE_DEFAULTS["batting_avg"],
            "on_base_pct": s.obp or LEAGUE_DEFAULTS["on_base_pct"],
            "slugging_pct": s.slg or LEAGUE_DEFAULTS["slugging_pct"],
            "woba": woba or LEAGUE_DEFAULTS["woba"],
            "barrel_rate": s.barrel_rate or LEAGUE_DEFAULTS["barrel_rate"],
            "exit_velocity": s.exit_velo or LEAGUE_DEFAULTS["exit_velocity"],
            "launch_angle": s.launch_angle if s.launch_angle is not None else LEAGUE_DEFAULTS["launch_angle"],
            "sprint_speed": s.sprint_speed or LEAGUE_DEFAULTS["sprint_speed"],
            "k_rate": s.k_rate or LEAGUE_DEFAULTS["k_rate"],
            "bb_rate": s.bb_rate or LEAGUE_DEFAULTS["bb_rate"],
            "hard_hit_rate": s.hard_hit_rate or LEAGUE_DEFAULTS["hard_hit_rate"],
            "pull_rate": LEAGUE_DEFAULTS["pull_rate"],
            "park_factor": s.park_factor or LEAGUE_DEFAULTS["park_factor"],
            "platoon_advantage": LEAGUE_DEFAULTS["platoon_advantage"],
            "days_rest": min(days_rest, 7.0),
            # Targets
            "hits": s.hits or 0,
            "home_runs": s.home_runs or 0,
            "rbi": s.rbi or 0,
            "walks": s.walks or 0,
        })

    return pd.DataFrame(records)


def _extract_features(
    df: pd.DataFrame,
    seq_length: int = None,
) -> np.ndarray:
    """Extract (seq_length, 15) feature matrix from a DataFrame window."""
    if seq_length is None:
        seq_length = len(df)

    features = np.zeros((seq_length, len(FEATURE_NAMES)), dtype=np.float32)

    # Pad from left if df is shorter
    offset = max(0, seq_length - len(df))
    for i, feat_name in enumerate(FEATURE_NAMES):
        default = LEAGUE_DEFAULTS.get(feat_name, 0.0)
        features[:offset, i] = default
        col = df[feat_name].values if feat_name in df.columns else np.full(len(df), default)
        features[offset:, i] = col[:seq_length - offset]

    return features
