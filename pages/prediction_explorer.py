"""
Prediction Explorer Page

Select a player (preset, custom, or live API), get predictions
with natural-language explanations and radar chart visualization.
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go

from src.models.model_registry import get_registry
from src.data.synthetic import (
    PRESET_PLAYERS, PLAYER_ARCHETYPES, TARGET_NAMES,
    FEATURE_NAMES, generate_player_sequence,
)
from src.data.feature_engineering import stats_to_game_log, game_logs_to_tensor


def show():
    st.title("Prediction Explorer")
    st.markdown("Select a player profile, get a prediction, and see a natural-language explanation.")

    registry = get_registry(st.session_state)

    if not registry.is_trained or "LSTM" not in registry.models:
        st.warning("Train models first on the **Model Comparison** page.")
        return

    model = registry.models["LSTM"].model

    # --- Player Selection ---
    mode = st.radio("Player Source", ["Preset Player", "Custom Stats"], horizontal=True)

    if mode == "Preset Player":
        player_name = st.selectbox("Select Player", list(PRESET_PLAYERS.keys()))
        player_info = PRESET_PLAYERS[player_name]
        X, _ = generate_player_sequence(
            archetype=player_info["archetype"],
            hot_streak_prob=player_info["hot_streak_prob"],
            cold_streak_prob=player_info["cold_streak_prob"],
            seed=hash(player_name) % 10000,
        )
        st.caption(f"Archetype: **{player_info['archetype'].title()}**")
    else:
        player_name = "Custom Player"
        X = _custom_stats_input()

    # --- Run Prediction ---
    sample = torch.FloatTensor(X)
    model.eval()
    with torch.no_grad():
        output, attn_weights = model.forward_with_attention(sample)

    pred = output[0].numpy()

    # --- Results ---
    st.markdown("---")
    st.subheader(f"Prediction for {player_name}")

    col1, col2 = st.columns([1, 1])

    with col1:
        _show_prediction_cards(pred)
        _show_explanation(pred, X[0], player_name)

    with col2:
        _show_prediction_radar(pred, player_name)


def _custom_stats_input() -> np.ndarray:
    """Sliders for custom player stats."""
    st.subheader("Custom Player Stats")
    col1, col2, col3 = st.columns(3)

    with col1:
        avg = st.slider("Batting Average", 0.150, 0.380, 0.265, 0.005)
        obp = st.slider("On-Base Pct", 0.200, 0.450, 0.330, 0.005)
        slg = st.slider("Slugging Pct", 0.250, 0.700, 0.430, 0.010)
        barrel = st.slider("Barrel Rate", 0.01, 0.20, 0.07, 0.01)
        exit_velo = st.slider("Exit Velocity", 82.0, 96.0, 88.5, 0.5)

    with col2:
        launch = st.slider("Launch Angle", 5.0, 25.0, 12.5, 0.5)
        speed = st.slider("Sprint Speed", 24.0, 31.0, 27.5, 0.5)
        k_rate = st.slider("K Rate", 0.05, 0.40, 0.22, 0.01)
        bb_rate = st.slider("BB Rate", 0.02, 0.20, 0.08, 0.01)
        hard_hit = st.slider("Hard Hit Rate", 0.20, 0.55, 0.35, 0.01)

    with col3:
        pull = st.slider("Pull Rate", 0.25, 0.55, 0.40, 0.01)
        park = st.slider("Park Factor", 0.80, 1.20, 1.00, 0.02)
        platoon = st.selectbox("Platoon Advantage", [0.0, 0.5, 1.0],
                                format_func=lambda x: {0.0: "None", 0.5: "Slight", 1.0: "Full"}[x])
        rest = st.slider("Days Rest", 0, 5, 1)

    game_log = stats_to_game_log(
        batting_avg=avg, on_base_pct=obp, slugging_pct=slg,
        barrel_rate=barrel, exit_velocity=exit_velo, launch_angle=launch,
        sprint_speed=speed, k_rate=k_rate, bb_rate=bb_rate,
        hard_hit_rate=hard_hit, pull_rate=pull, park_factor=park,
        platoon_advantage=platoon, days_rest=float(rest),
    )

    # Create a sequence of 10 similar games (slight variation)
    rng = np.random.RandomState(0)
    logs = []
    for t in range(10):
        log = dict(game_log)
        for key in ["batting_avg", "on_base_pct", "slugging_pct"]:
            log[key] += rng.randn() * 0.015
        logs.append(log)
    logs[-1] = game_log  # latest game is the exact input

    return game_logs_to_tensor(logs)


def _show_prediction_cards(pred: np.ndarray):
    """Display prediction values as metric cards."""
    cols = st.columns(len(TARGET_NAMES))
    formats = {"hits": "1.2f", "home_runs": "1.3f", "rbi": "1.2f", "walks": "1.2f"}

    for i, (col, name) in enumerate(zip(cols, TARGET_NAMES)):
        with col:
            fmt = formats.get(name, "1.3f")
            label = name.replace("_", " ").title()
            st.metric(label, f"{pred[i]:{fmt}}")


def _show_explanation(pred: np.ndarray, features: np.ndarray, player_name: str):
    """Generate a natural-language explanation of the prediction."""
    st.subheader("Explanation")

    hits, hr, rbi, walks = pred
    recent = features[-1]  # last timestep
    avg = recent[0]
    slg = recent[2]
    k_rate = recent[8]
    bb_rate = recent[9]
    exit_velo = recent[5]

    lines = []

    # Overall assessment
    if hits > 1.3:
        lines.append(f"**{player_name}** is projected for a strong multi-hit game ({hits:.1f} hits expected).")
    elif hits > 0.8:
        lines.append(f"**{player_name}** is projected for a typical game ({hits:.1f} hits expected).")
    else:
        lines.append(f"**{player_name}** may struggle at the plate today ({hits:.1f} hits expected).")

    # Power assessment
    if hr > 0.15:
        lines.append(f"There's meaningful home run upside ({hr:.2f} expected HR) driven by "
                     f"a {slg:.3f} slugging percentage and {exit_velo:.1f} mph exit velocity.")
    else:
        lines.append(f"Home run probability is low ({hr:.2f}) for this matchup.")

    # Discipline
    if bb_rate > 0.10:
        lines.append(f"Strong plate discipline ({bb_rate:.1%} walk rate) suggests "
                     f"{walks:.1f} walks expected.")
    elif k_rate > 0.25:
        lines.append(f"Elevated strikeout rate ({k_rate:.1%}) may limit production.")

    # RBI opportunity
    if rbi > 0.8:
        lines.append(f"With {rbi:.1f} RBI projected, run-producing opportunities look favorable.")

    st.markdown("\n\n".join(lines))


def _show_prediction_radar(pred: np.ndarray, player_name: str):
    """Radar chart of prediction vs league average."""
    st.subheader("Prediction Profile")

    # League averages for reference
    league_avg = np.array([1.0, 0.08, 0.55, 0.35])

    categories = ["Hits", "Home Runs", "RBI", "Walks"]

    # Normalize to 0-1 scale
    max_vals = np.array([3.0, 0.5, 3.0, 2.0])
    pred_norm = np.clip(pred / max_vals, 0, 1)
    avg_norm = league_avg / max_vals

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(avg_norm) + [avg_norm[0]],
        theta=categories + [categories[0]],
        name="League Avg",
        line=dict(color="gray", dash="dash"),
    ))

    fig.add_trace(go.Scatterpolar(
        r=list(pred_norm) + [pred_norm[0]],
        theta=categories + [categories[0]],
        name=player_name,
        fill="toself",
        line=dict(color="#636EFA"),
        opacity=0.4,
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
