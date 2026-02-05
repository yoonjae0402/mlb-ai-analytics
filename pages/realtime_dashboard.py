"""
Real-Time Dashboard Page

Live scoreboard with win probability tracking.
Demo mode with "Advance Inning" simulation when no live games.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.services.realtime import fetch_live_games, advance_inning


def show():
    st.title("Real-Time Dashboard")

    # Initialize game state
    if "rt_games" not in st.session_state:
        games, mode = fetch_live_games()
        st.session_state["rt_games"] = games
        st.session_state["rt_mode"] = mode

    mode = st.session_state["rt_mode"]
    games = st.session_state["rt_games"]

    # Mode banner
    if mode == "live":
        st.success("Connected to MLB Stats API — showing live games.")
    elif mode == "schedule":
        st.info("No live games right now. Showing today's schedule.")
    else:
        st.info("Demo Mode — simulated games. Click **Advance Inning** to progress.")

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        if st.button("Refresh Games", use_container_width=True):
            games, mode = fetch_live_games()
            st.session_state["rt_games"] = games
            st.session_state["rt_mode"] = mode
            st.rerun()
    with col_ctrl2:
        if mode == "demo":
            if st.button("Advance Inning", type="primary", use_container_width=True):
                st.session_state["rt_games"] = [advance_inning(g) for g in games]
                st.rerun()

    st.markdown("---")

    # --- Scoreboard ---
    _show_scoreboard(games)

    st.markdown("---")

    # --- Win Probability Chart ---
    if any(len(g.get("wp_history", [])) > 1 for g in games):
        _show_win_probability(games)


def _show_scoreboard(games: list[dict]):
    """Display game scoreboard cards."""
    st.subheader("Scoreboard")

    cols_per_row = 3
    for row_start in range(0, len(games), cols_per_row):
        row_games = games[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, game in zip(cols, row_games):
            with col:
                status = game["status"]
                inning_str = ""
                if status == "In Progress":
                    inning_str = f" | {game['half']} {game['inning']}"
                elif status == "Final":
                    inning_str = " | Final"
                elif status == "Scheduled":
                    inning_str = " | Scheduled"

                st.markdown(f"**{game['away_abbrev']}** {game['away_score']} - "
                            f"{game['home_score']} **{game['home_abbrev']}**{inning_str}")

                if status not in ("Scheduled",):
                    wp = game.get("home_win_prob", 0.5)
                    bar_color = "#00CC96" if wp > 0.5 else "#EF553B"
                    st.progress(wp, text=f"{game['home_abbrev']} Win Prob: {wp:.0%}")


def _show_win_probability(games: list[dict]):
    """Win probability area chart for selected game."""
    st.subheader("Win Probability Tracker")

    active_games = [g for g in games if len(g.get("wp_history", [])) > 1]
    if not active_games:
        st.caption("No win probability data yet. Advance innings to generate.")
        return

    game_labels = [f"{g['away_abbrev']} @ {g['home_abbrev']}" for g in active_games]
    selected_idx = st.selectbox("Select game", range(len(active_games)),
                                 format_func=lambda i: game_labels[i])

    game = active_games[selected_idx]
    wp = game["wp_history"]
    innings = list(range(len(wp)))

    fig = go.Figure()

    # Fill above 0.5 (home favored)
    fig.add_trace(go.Scatter(
        x=innings, y=wp,
        mode="lines+markers",
        name=f"{game['home_abbrev']} Win Prob",
        line=dict(color="#636EFA", width=3),
        fill="tonexty" if False else None,
    ))

    # 50% reference line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="50/50", annotation_position="bottom right")

    # Fill regions
    home_above = [max(w, 0.5) for w in wp]
    home_below = [min(w, 0.5) for w in wp]

    fig.add_trace(go.Scatter(
        x=innings, y=home_above,
        fill="tonexty", fillcolor="rgba(99,110,250,0.2)",
        line=dict(width=0), showlegend=False,
        name="Home favored",
    ))
    fig.add_trace(go.Scatter(
        x=innings, y=[0.5] * len(innings),
        line=dict(width=0), showlegend=False,
    ))

    fig.update_layout(
        xaxis_title="Half-Inning",
        yaxis_title="Home Win Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=400,
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Game summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Home Win Prob", f"{wp[-1]:.0%}")
    with col2:
        max_swing = max(abs(wp[i] - wp[i-1]) for i in range(1, len(wp))) if len(wp) > 1 else 0
        st.metric("Max WP Swing", f"{max_swing:.0%}")
    with col3:
        st.metric("Innings Played", f"{game['inning']}")
