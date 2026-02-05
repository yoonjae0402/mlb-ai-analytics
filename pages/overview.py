"""Overview page - Interactive landing page with quick-start feature cards."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.models.model_registry import get_registry


def show():
    # Hero section
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <h1 style="font-size:2.5rem; margin-bottom:0.2rem;">MLB AI Analytics Platform</h1>
        <p style="font-size:1.15rem; color:#666;">
            Train models, visualize attention, build ensembles, and explore predictions &mdash; all interactive.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Status banner
    registry = get_registry(st.session_state)
    if registry.is_trained:
        n_models = len(registry.models)
        st.success(f"{n_models} model(s) trained and ready. Explore the pages below.")
    else:
        st.info("Start by training models on the **Model Comparison** page.")

    st.markdown("---")

    # Quick-start feature cards
    st.subheader("Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="tech-card">
            <h3>Model Comparison</h3>
            <p>Train LSTM and XGBoost side-by-side. Compare training curves, scatter plots, and per-target metrics.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tech-card">
            <h3>Attention Visualizer</h3>
            <p>See what the LSTM attends to via heatmaps, temporal importance, and gradient-based feature attribution.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="tech-card">
            <h3>Ensemble Lab</h3>
            <p>Blend models with tunable weights or stacking. Find the optimal mix with weight sensitivity curves.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tech-card">
            <h3>Prediction Explorer</h3>
            <p>Pick a player, get a prediction with natural-language explanation and radar chart visualization.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="tech-card">
            <h3>Real-Time Dashboard</h3>
            <p>Live scoreboard with win probability tracking. Includes demo mode with simulated game progression.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tech-card">
            <h3>Architecture & Docs</h3>
            <p>System architecture diagram, tech stack reference, and collapsible code walkthroughs.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Sample visualization
    st.subheader("Sample: Player Archetype Profiles")
    _show_archetype_radar()


def _show_archetype_radar():
    """Radar chart showing different player archetype stat profiles."""
    from src.data.synthetic import PLAYER_ARCHETYPES

    categories = ["AVG", "OBP", "SLG", "wOBA", "Barrel%", "K% (inv)", "BB%", "Speed"]

    fig = go.Figure()
    colors = {"power": "#636EFA", "contact": "#EF553B", "balanced": "#00CC96", "speedster": "#AB63FA"}

    for name, arch in PLAYER_ARCHETYPES.items():
        base = arch["base_mean"]
        # Normalize each stat to 0-1 range for radar
        vals = [
            base[0] / 0.350,       # AVG
            base[1] / 0.420,       # OBP
            base[2] / 0.600,       # SLG
            base[3] / 0.420,       # wOBA
            base[4] / 0.15,        # barrel_rate
            1.0 - base[8] / 0.35,  # k_rate (inverted - lower is better)
            base[9] / 0.15,        # bb_rate
            (base[7] - 25) / 6,    # sprint_speed normalized
        ]
        vals.append(vals[0])  # close the polygon

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            name=name.title(),
            line=dict(color=colors.get(name, "#999")),
            fill="toself",
            opacity=0.3,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
