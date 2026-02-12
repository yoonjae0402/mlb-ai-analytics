"""Feature engineering from real MLB player stats.

Builds (N, seq_length, num_features) feature arrays from player_stats rows,
with rolling averages, trend features, pitcher matchup data, and contextual data.
Includes StandardScaler for feature normalization.
"""

import logging
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Global scaler — fitted during training, reused during inference
_scaler: Optional[StandardScaler] = None
_scaler_path = Path("models/feature_scaler.pkl")

# 22 features: 15 original batter + 7 pitcher/context
FEATURE_NAMES = [
    # Batter core (15)
    "batting_avg", "on_base_pct", "slugging_pct", "woba",
    "barrel_rate", "exit_velocity", "launch_angle", "sprint_speed",
    "k_rate", "bb_rate", "hard_hit_rate", "pull_rate",
    "park_factor", "platoon_advantage", "days_rest",
    # Pitcher matchup (5)
    "opp_era", "opp_whip", "opp_k_per_9", "opp_bb_per_9", "opp_handedness_adv",
    # Derived context (2)
    "iso", "hot_streak",
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
    # Pitcher defaults (league average)
    "opp_era": 4.20,
    "opp_whip": 1.30,
    "opp_k_per_9": 8.5,
    "opp_bb_per_9": 3.2,
    "opp_handedness_adv": 0.0,
    # Derived
    "iso": 0.166,  # SLG - BA
    "hot_streak": 0.0,
}

NUM_FEATURES = len(FEATURE_NAMES)


def get_scaler() -> Optional[StandardScaler]:
    """Get the fitted scaler, loading from disk if needed."""
    global _scaler
    if _scaler is not None:
        return _scaler
    if _scaler_path.exists():
        with open(_scaler_path, "rb") as f:
            _scaler = pickle.load(f)
        logger.info(f"Loaded feature scaler from {_scaler_path}")
    return _scaler


def fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training data and persist it.

    Args:
        X: (N, seq_length, num_features) training features

    Returns:
        Fitted StandardScaler
    """
    global _scaler
    # Reshape to 2D for fitting: (N * seq_length, num_features)
    n, seq_len, n_feat = X.shape
    X_flat = X.reshape(-1, n_feat)

    scaler = StandardScaler()
    scaler.fit(X_flat)

    _scaler = scaler
    _scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Fitted and saved feature scaler to {_scaler_path} ({n_feat} features)")
    return scaler


def scale_features(X: np.ndarray) -> np.ndarray:
    """Apply StandardScaler to feature sequences.

    Args:
        X: (N, seq_length, num_features) or (1, seq_length, num_features)

    Returns:
        Scaled array of same shape
    """
    scaler = get_scaler()
    if scaler is None:
        logger.warning("No fitted scaler found — returning unscaled features")
        return X

    original_shape = X.shape
    n_feat = X.shape[-1]

    # Handle feature count mismatch (model trained with different feature count)
    if scaler.n_features_in_ != n_feat:
        logger.warning(
            f"Scaler expects {scaler.n_features_in_} features but got {n_feat} — skipping scaling"
        )
        return X

    X_flat = X.reshape(-1, n_feat)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(original_shape).astype(np.float32)


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
        X: (N, seq_length, num_features) feature sequences (unscaled)
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

    # Load pitcher stats lookup
    pitcher_lookup = _build_pitcher_lookup(session, seasons)

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
        df = _stats_to_dataframe(stats, pitcher_lookup)

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

    logger.info(f"Built {X.shape[0]} sequences ({X.shape[2]} features) from {len(by_player)} players")
    return X, y, pid_list


def build_player_sequence(
    session,
    player_id: int,
    seq_length: int = 10,
) -> Optional[np.ndarray]:
    """Build a single feature sequence for a player (most recent games).

    Returns:
        (1, seq_length, num_features) scaled array or None if insufficient data.
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
    X = features.reshape(1, seq_length, -1)

    # Apply scaling for inference
    return scale_features(X)


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    player_ids: list[int],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict:
    """Time-based split to prevent data leakage.

    Fits the scaler on training data only, then scales all splits.
    """
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    # Fit scaler on training data only (prevents data leakage)
    fit_scaler(X_train)

    # Scale all splits
    X_train = scale_features(X_train)
    X_val = scale_features(X_val)
    X_test = scale_features(X_test)

    return {
        "X_train": X_train,
        "y_train": y[:train_end],
        "X_val": X_val,
        "y_val": y[train_end:val_end],
        "X_test": X_test,
        "y_test": y[val_end:],
        "train_player_ids": player_ids[:train_end],
        "val_player_ids": player_ids[train_end:val_end],
        "test_player_ids": player_ids[val_end:],
    }


def _build_pitcher_lookup(session, seasons: list[int] = None) -> dict:
    """Build a lookup of pitcher stats by (team, date) for opponent matching.

    Returns dict mapping (team, date) -> {era, whip, k_per_9, bb_per_9, throws}
    Uses actual pitching game logs (innings_pitched, earned_runs, etc.).
    """
    try:
        from backend.db.models import Player, PlayerStat
        from sqlalchemy import extract

        query = (
            session.query(
                Player.team, PlayerStat.game_date,
                PlayerStat.innings_pitched, PlayerStat.earned_runs,
                PlayerStat.k_rate, PlayerStat.bb_rate,
                PlayerStat.strikeouts, PlayerStat.walks, PlayerStat.hits,
                Player.throws,
            )
            .join(PlayerStat)
            .filter(Player.position == "P")
        )
        if seasons:
            query = query.filter(extract("year", PlayerStat.game_date).in_(seasons))

        results = query.all()
        if not results:
            return {}

        from collections import defaultdict
        team_date_stats = defaultdict(list)
        for (team, gdate, ip, er, k_rate, bb_rate,
             k, bb, h, throws) in results:
            if not team:
                continue
            ip = ip or 0.0
            team_date_stats[(team, gdate)].append({
                "ip": ip,
                "er": er or 0,
                "k": k or 0,
                "bb": bb or 0,
                "h": h or 0,
                "k_rate": k_rate,
                "bb_rate": bb_rate,
                "throws": throws or "R",
            })

        lookup = {}
        for (team, gdate), stats_list in team_date_stats.items():
            total_ip = sum(s["ip"] for s in stats_list)
            total_er = sum(s["er"] for s in stats_list)
            total_k = sum(s["k"] for s in stats_list)
            total_bb = sum(s["bb"] for s in stats_list)
            total_h = sum(s["h"] for s in stats_list)

            # Compute ERA and WHIP from game-level data
            era = (total_er / total_ip * 9) if total_ip > 0 else LEAGUE_DEFAULTS["opp_era"]
            whip = ((total_bb + total_h) / total_ip) if total_ip > 0 else LEAGUE_DEFAULTS["opp_whip"]

            # K/9 and BB/9: prefer stored rates, fall back to computed
            k_rates = [s["k_rate"] for s in stats_list if s["k_rate"] is not None]
            bb_rates = [s["bb_rate"] for s in stats_list if s["bb_rate"] is not None]
            k_per_9 = np.mean(k_rates) if k_rates else (total_k / total_ip * 9 if total_ip > 0 else LEAGUE_DEFAULTS["opp_k_per_9"])
            bb_per_9 = np.mean(bb_rates) if bb_rates else (total_bb / total_ip * 9 if total_ip > 0 else LEAGUE_DEFAULTS["opp_bb_per_9"])

            hands = [s["throws"] for s in stats_list]
            dominant_hand = max(set(hands), key=hands.count) if hands else "R"

            lookup[(team, gdate)] = {
                "era": era,
                "whip": whip,
                "k_per_9": k_per_9,
                "bb_per_9": bb_per_9,
                "throws": dominant_hand,
            }

        logger.info(f"Built pitcher lookup: {len(lookup)} team-date entries")
        return lookup
    except Exception as e:
        logger.warning(f"Could not build pitcher lookup: {e}")
        return {}


def _stats_to_dataframe(stats: list, pitcher_lookup: dict = None) -> pd.DataFrame:
    """Convert a list of PlayerStat ORM objects to a DataFrame."""
    records = []
    prev_date = None
    recent_hits = []  # for hot streak calculation

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

        ba = s.batting_avg or LEAGUE_DEFAULTS["batting_avg"]
        slg = s.slg or LEAGUE_DEFAULTS["slugging_pct"]

        # Derived: ISO (isolated power) = SLG - BA
        iso = max(slg - ba, 0.0)

        # Hot streak: 1 if batting avg > .350 over last 7 games
        ab = s.at_bats or 0
        hits = s.hits or 0
        recent_hits.append((hits, ab))
        if len(recent_hits) > 7:
            recent_hits = recent_hits[-7:]
        total_h = sum(h for h, _ in recent_hits)
        total_ab = sum(a for _, a in recent_hits)
        hot_streak = 1.0 if total_ab >= 15 and total_h / total_ab > 0.350 else 0.0

        # Pitcher matchup features (best-effort from lookup)
        opp_era = LEAGUE_DEFAULTS["opp_era"]
        opp_whip = LEAGUE_DEFAULTS["opp_whip"]
        opp_k_per_9 = LEAGUE_DEFAULTS["opp_k_per_9"]
        opp_bb_per_9 = LEAGUE_DEFAULTS["opp_bb_per_9"]
        opp_handedness_adv = LEAGUE_DEFAULTS["opp_handedness_adv"]

        if pitcher_lookup:
            # Match opposing pitcher data by scanning all teams on the same date
            # (since we don't store opponent team directly on PlayerStat)
            for (team, pdate), p_stats in pitcher_lookup.items():
                if pdate == s.game_date:
                    # Skip same team (we want opponent pitchers)
                    batter_player = getattr(s, 'player', None)
                    batter_team = batter_player.team if batter_player else None
                    if batter_team and team == batter_team:
                        continue
                    # Found an opposing team's pitching data for this date
                    opp_era = p_stats.get("era", opp_era)
                    opp_whip = p_stats.get("whip", opp_whip)
                    opp_k_per_9 = p_stats.get("k_per_9", opp_k_per_9)
                    opp_bb_per_9 = p_stats.get("bb_per_9", opp_bb_per_9)
                    # Handedness advantage
                    batter_bats = batter_player.bats if batter_player else ''
                    pitcher_throws = p_stats.get("throws", "R")
                    if batter_bats and pitcher_throws:
                        opp_handedness_adv = 1.0 if batter_bats != pitcher_throws else -0.5
                    break  # Use first opposing team match

        records.append({
            "game_date": s.game_date,
            "batting_avg": ba,
            "on_base_pct": s.obp or LEAGUE_DEFAULTS["on_base_pct"],
            "slugging_pct": slg,
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
            # Pitcher matchup
            "opp_era": opp_era,
            "opp_whip": opp_whip,
            "opp_k_per_9": opp_k_per_9,
            "opp_bb_per_9": opp_bb_per_9,
            "opp_handedness_adv": opp_handedness_adv,
            # Derived
            "iso": iso,
            "hot_streak": hot_streak,
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
    """Extract (seq_length, num_features) feature matrix from a DataFrame window."""
    if seq_length is None:
        seq_length = len(df)

    features = np.zeros((seq_length, NUM_FEATURES), dtype=np.float32)

    # Pad from left if df is shorter
    offset = max(0, seq_length - len(df))
    for i, feat_name in enumerate(FEATURE_NAMES):
        default = LEAGUE_DEFAULTS.get(feat_name, 0.0)
        features[:offset, i] = default
        col = df[feat_name].values if feat_name in df.columns else np.full(len(df), default)
        features[offset:, i] = col[:seq_length - offset]

    return features
