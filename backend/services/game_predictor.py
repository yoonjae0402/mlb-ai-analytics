"""Game-level prediction service.

Aggregates player-level predictions into projected team runs
and computes win probability using Pythagorean expectation.
"""

import logging
import math
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from backend.db.models import Player, Prediction, Team

logger = logging.getLogger(__name__)

# MLB average runs per game (2023-2024 seasons)
LEAGUE_AVG_RUNS = 4.5
# Pythagorean exponent — empirically ~1.83 for MLB
PYTH_EXPONENT = 1.83


@dataclass
class TeamProjection:
    team_name: str
    team_abbreviation: str
    projected_runs: float
    projected_hits: float
    projected_hr: float
    projected_rbi: float
    projected_walks: float
    n_players_with_predictions: int
    n_total_players: int


@dataclass
class WinProbabilityResult:
    home_win_pct: float
    away_win_pct: float
    home: TeamProjection
    away: TeamProjection
    confidence: float
    method: str


def _get_team_abbreviation(session: Session, mlb_team_id: int, fallback: str) -> str:
    """Resolve MLB team ID to abbreviation."""
    if mlb_team_id:
        team = session.query(Team).filter_by(mlb_id=mlb_team_id).first()
        if team:
            return team.abbreviation
    return fallback


def _get_team_players_with_predictions(
    session: Session, team_abbr: str, team_name_fallback: str
) -> tuple[list[dict], int]:
    """Get players for a team with their latest predictions.

    Returns (list of player prediction dicts, total player count).
    """
    from sqlalchemy import or_

    players = (
        session.query(Player)
        .filter(
            or_(
                Player.team == team_abbr,
                Player.team.ilike(f"%{team_name_fallback}%"),
            )
        )
        .filter(Player.position != "P")  # Only batters
        .all()
    )

    total = len(players)
    results = []

    for p in players:
        pred = (
            session.query(Prediction)
            .filter(Prediction.player_id == p.id)
            .order_by(desc(Prediction.created_at))
            .first()
        )
        if pred and pred.predicted_hits is not None:
            results.append({
                "player_id": p.id,
                "name": p.name,
                "predicted_hits": pred.predicted_hits or 0,
                "predicted_hr": pred.predicted_hr or 0,
                "predicted_rbi": pred.predicted_rbi or 0,
                "predicted_walks": pred.predicted_walks or 0,
                "confidence": pred.confidence or 0.5,
            })

    return results, total


def _estimate_runs_from_predictions(player_preds: list[dict]) -> float:
    """Estimate team runs from individual player predictions.

    Uses a weighted combination:
    - Predicted hits contribute to run scoring
    - HRs are direct runs (at least 1 run each)
    - RBI is the most direct run measure
    - Walks contribute to baserunners

    Simple model: projected_runs ≈ 0.5 * total_hits + 1.4 * total_hr + 0.3 * total_walks
    Calibrated to produce ~4-5 runs per game for an average lineup of 9 batters.
    """
    if not player_preds:
        return LEAGUE_AVG_RUNS

    total_hits = sum(p["predicted_hits"] for p in player_preds)
    total_hr = sum(p["predicted_hr"] for p in player_preds)
    total_walks = sum(p["predicted_walks"] for p in player_preds)
    total_rbi = sum(p["predicted_rbi"] for p in player_preds)

    # Primary method: use RBI as the core run estimate
    # RBI directly measures runs produced, but double-counts HR
    # Blend RBI with a component model for robustness
    component_runs = 0.5 * total_hits + 1.4 * total_hr + 0.3 * total_walks
    rbi_runs = total_rbi

    # Weighted blend: 60% RBI-based, 40% component-based
    estimated = 0.6 * rbi_runs + 0.4 * component_runs

    # Scale by lineup coverage (if we only have predictions for 5 of 9 batters)
    n_batters = len(player_preds)
    if n_batters < 9:
        # Extrapolate to full lineup, but conservatively
        scale = 9.0 / max(n_batters, 1)
        estimated *= min(scale, 2.0)  # Cap at 2x to avoid wild extrapolations

    return round(estimated, 2)


def _pythagorean_win_pct(
    team_runs: float, opponent_runs: float, exponent: float = PYTH_EXPONENT
) -> float:
    """Pythagorean expectation: RS^exp / (RS^exp + RA^exp).

    This is the standard baseball win probability formula
    used by Baseball Prospectus and FanGraphs.
    """
    if team_runs <= 0 and opponent_runs <= 0:
        return 0.5
    if opponent_runs <= 0:
        return 0.95  # Cap at 95%
    if team_runs <= 0:
        return 0.05  # Floor at 5%

    rs_exp = team_runs ** exponent
    ra_exp = opponent_runs ** exponent
    return rs_exp / (rs_exp + ra_exp)


def compute_win_probability(
    session: Session,
    home_team_id: int,
    away_team_id: int,
    home_team_name: str,
    away_team_name: str,
) -> WinProbabilityResult:
    """Compute game-level win probability from player predictions.

    1. Look up team players and their latest predictions
    2. Aggregate to projected team runs
    3. Apply Pythagorean expectation for win probability
    """
    home_abbr = _get_team_abbreviation(session, home_team_id, home_team_name)
    away_abbr = _get_team_abbreviation(session, away_team_id, away_team_name)

    home_preds, home_total = _get_team_players_with_predictions(
        session, home_abbr, home_team_name
    )
    away_preds, away_total = _get_team_players_with_predictions(
        session, away_abbr, away_team_name
    )

    # Aggregate
    home_runs = _estimate_runs_from_predictions(home_preds)
    away_runs = _estimate_runs_from_predictions(away_preds)

    # Home field advantage: ~54% historical MLB average
    # Add ~0.3 runs to home team projection
    home_runs_adj = home_runs + 0.3

    home_wp = _pythagorean_win_pct(home_runs_adj, away_runs)
    away_wp = 1.0 - home_wp

    # Confidence: based on prediction coverage
    home_coverage = len(home_preds) / max(home_total, 1)
    away_coverage = len(away_preds) / max(away_total, 1)
    avg_confidence = (home_coverage + away_coverage) / 2.0

    # Also factor in individual prediction confidences
    all_conf = [p["confidence"] for p in home_preds + away_preds]
    if all_conf:
        pred_confidence = sum(all_conf) / len(all_conf)
        overall_confidence = 0.5 * avg_confidence + 0.5 * pred_confidence
    else:
        overall_confidence = 0.3  # Low confidence if no predictions

    home_proj = TeamProjection(
        team_name=home_team_name,
        team_abbreviation=home_abbr,
        projected_runs=home_runs_adj,
        projected_hits=sum(p["predicted_hits"] for p in home_preds),
        projected_hr=sum(p["predicted_hr"] for p in home_preds),
        projected_rbi=sum(p["predicted_rbi"] for p in home_preds),
        projected_walks=sum(p["predicted_walks"] for p in home_preds),
        n_players_with_predictions=len(home_preds),
        n_total_players=home_total,
    )

    away_proj = TeamProjection(
        team_name=away_team_name,
        team_abbreviation=away_abbr,
        projected_runs=away_runs,
        projected_hits=sum(p["predicted_hits"] for p in away_preds),
        projected_hr=sum(p["predicted_hr"] for p in away_preds),
        projected_rbi=sum(p["predicted_rbi"] for p in away_preds),
        projected_walks=sum(p["predicted_walks"] for p in away_preds),
        n_players_with_predictions=len(away_preds),
        n_total_players=away_total,
    )

    return WinProbabilityResult(
        home_win_pct=round(home_wp, 3),
        away_win_pct=round(away_wp, 3),
        home=home_proj,
        away=away_proj,
        confidence=round(overall_confidence, 3),
        method="pythagorean_expectation",
    )
