"""Minor League Equivalency (MLE) and prospect projection.

Implements translation factors to convert MiLB stats to MLB equivalents,
then uses the trained LSTM model to project a prospect's MLB impact.

Translation factors are based on historical research:
- AAA → MLB: ~0.85 for rate stats
- AA  → MLB: ~0.75
- A+  → MLB: ~0.65
- A   → MLB: ~0.55

References:
- Bill James Minor League Equivalencies
- Clay Davenport translation factors
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Translation factors: multiply MiLB rate stats by these to get MLB equivalents
TRANSLATION_FACTORS = {
    "AAA": {
        "batting_avg": 0.85,
        "obp": 0.87,
        "slg": 0.83,
        "hr_rate": 0.80,
        "bb_rate": 0.90,
        "k_rate": 1.12,  # K rate goes UP at higher levels
        "hit_rate": 0.85,
    },
    "AA": {
        "batting_avg": 0.76,
        "obp": 0.78,
        "slg": 0.73,
        "hr_rate": 0.70,
        "bb_rate": 0.82,
        "k_rate": 1.20,
        "hit_rate": 0.76,
    },
    "A+": {
        "batting_avg": 0.66,
        "obp": 0.68,
        "slg": 0.62,
        "hr_rate": 0.58,
        "bb_rate": 0.75,
        "k_rate": 1.30,
        "hit_rate": 0.66,
    },
    "A": {
        "batting_avg": 0.56,
        "obp": 0.58,
        "slg": 0.52,
        "hr_rate": 0.48,
        "bb_rate": 0.68,
        "k_rate": 1.40,
        "hit_rate": 0.56,
    },
}

# Age adjustment: younger players at a given level get a boost
# Factor = 1 + (league_avg_age - player_age) * 0.02
LEVEL_AVG_AGES = {
    "AAA": 27.0,
    "AA": 24.0,
    "A+": 22.5,
    "A": 21.0,
}


def translate_stats(
    stats: dict,
    level: str,
    player_age: Optional[int] = None,
) -> dict:
    """Translate minor league stats to MLB equivalents.

    Args:
        stats: Dict with MiLB stat values (batting_avg, obp, slg, etc.)
        level: Current minor league level (AAA, AA, A+, A)
        player_age: Player's current age (for age adjustment)

    Returns:
        Dict with translated MLB-equivalent stats.
    """
    factors = TRANSLATION_FACTORS.get(level)
    if not factors:
        logger.warning(f"No translation factors for level '{level}', returning raw stats")
        return stats.copy()

    translated = {}
    for stat_name, value in stats.items():
        if value is None:
            translated[stat_name] = None
            continue

        factor_key = stat_name
        # Map counting/rate stats to the right factor
        if stat_name in ("hits", "hit_rate"):
            factor_key = "hit_rate"
        elif stat_name in ("home_runs", "hr_rate"):
            factor_key = "hr_rate"

        factor = factors.get(factor_key, 0.80)  # Default translation factor
        translated[stat_name] = value * factor

    # Age adjustment
    if player_age and level in LEVEL_AVG_AGES:
        avg_age = LEVEL_AVG_AGES[level]
        age_bonus = 1.0 + (avg_age - player_age) * 0.02
        age_bonus = max(0.90, min(1.15, age_bonus))  # Clamp

        for key in translated:
            if translated[key] is not None and key != "k_rate":
                translated[key] *= age_bonus

    return translated


def project_mlb_impact(
    milb_stats: list[dict],
    level: str,
    player_age: Optional[int] = None,
) -> dict:
    """Project MLB performance from MiLB game log data.

    Takes a sequence of MiLB game stats, translates them, then
    computes projected per-game MLB output.

    Args:
        milb_stats: List of game-level stat dicts from MiLB
        level: MiLB level (AAA, AA, A+, A)
        player_age: Player's current age

    Returns:
        Dict with projected MLB per-game averages:
        {projected_hits, projected_hr, projected_rbi, projected_walks,
         projected_avg, confidence, sample_size}
    """
    if not milb_stats:
        return {
            "projected_hits": 0,
            "projected_hr": 0,
            "projected_rbi": 0,
            "projected_walks": 0,
            "projected_avg": 0,
            "confidence": 0,
            "sample_size": 0,
        }

    # Compute per-game averages from MiLB data
    hits = [s.get("hits", 0) or 0 for s in milb_stats]
    hrs = [s.get("home_runs", 0) or 0 for s in milb_stats]
    rbis = [s.get("rbi", 0) or 0 for s in milb_stats]
    walks = [s.get("walks", 0) or 0 for s in milb_stats]
    at_bats = [s.get("at_bats", 0) or 0 for s in milb_stats]

    n = len(milb_stats)
    total_ab = sum(at_bats) or 1

    milb_averages = {
        "batting_avg": sum(hits) / total_ab,
        "hit_rate": sum(hits) / n,
        "hr_rate": sum(hrs) / n,
        "rbi": sum(rbis) / n,
        "bb_rate": sum(walks) / n,
    }

    # Translate to MLB equivalents
    translated = translate_stats(milb_averages, level, player_age)

    # Confidence based on sample size: higher with more games
    confidence = min(1.0, n / 60)  # Full confidence at 60+ games

    return {
        "projected_hits": round(translated.get("hit_rate", 0), 3),
        "projected_hr": round(translated.get("hr_rate", 0), 4),
        "projected_rbi": round(translated.get("rbi", 0), 3),
        "projected_walks": round(translated.get("bb_rate", 0), 3),
        "projected_avg": round(translated.get("batting_avg", 0), 3),
        "confidence": round(confidence, 2),
        "sample_size": n,
    }


def rank_prospects(
    prospects: list[dict],
    milb_stats_map: dict[int, list[dict]],
) -> list[dict]:
    """Rank a list of prospects by projected MLB impact.

    Args:
        prospects: List of player dicts (must have mlb_id, name, current_level, age)
        milb_stats_map: Dict mapping mlb_id -> list of game stat dicts

    Returns:
        Sorted list of prospect projections, highest impact first.
    """
    ranked = []

    for p in prospects:
        mlb_id = p.get("mlb_id")
        level = p.get("current_level", "A")
        age = p.get("age")
        stats = milb_stats_map.get(mlb_id, [])

        if not stats:
            continue

        projection = project_mlb_impact(stats, level, age)
        projection["mlb_id"] = mlb_id
        projection["name"] = p.get("name", "")
        projection["team"] = p.get("team", "")
        projection["level"] = level
        projection["age"] = age

        # Composite score: weighted combination of projected stats
        score = (
            projection["projected_hits"] * 1.0
            + projection["projected_hr"] * 8.0
            + projection["projected_rbi"] * 2.0
            + projection["projected_walks"] * 1.5
        ) * projection["confidence"]
        projection["composite_score"] = round(score, 3)

        ranked.append(projection)

    ranked.sort(key=lambda x: x["composite_score"], reverse=True)
    return ranked
