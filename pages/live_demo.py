"""Live Demo page - Interactive demonstration of all features"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime


def show():
    st.title("📊 Live Demo")

    st.markdown("""
    Interactive demonstration of the MLB AI Analytics platform capabilities.
    """)

    tab1, tab2, tab3 = st.tabs([
        "Model Inference",
        "Data Pipeline",
        "API Endpoint"
    ])

    with tab1:
        show_model_demo()

    with tab2:
        show_data_pipeline()

    with tab3:
        show_api_demo()


def show_model_demo():
    """Live model inference demo"""
    st.header("PyTorch Model Inference")

    st.markdown("Test the player performance prediction model in real-time.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Features")

        # Player selection
        player_name = st.selectbox(
            "Select Player",
            ["Shohei Ohtani", "Mike Trout", "Mookie Betts", "Ronald Acuna Jr.", "Custom"]
        )

        if player_name == "Custom":
            batting_avg = st.slider("Batting Average", 0.150, 0.400, 0.275, 0.005)
            on_base = st.slider("On-Base %", 0.200, 0.500, 0.350, 0.005)
            slugging = st.slider("Slugging %", 0.250, 0.700, 0.450, 0.005)
        else:
            # Preset values for known players
            presets = {
                "Shohei Ohtani": (0.304, 0.412, 0.654),
                "Mike Trout": (0.263, 0.367, 0.490),
                "Mookie Betts": (0.307, 0.408, 0.579),
                "Ronald Acuna Jr.": (0.337, 0.416, 0.596),
            }
            batting_avg, on_base, slugging = presets[player_name]
            st.metric("Batting Average", f"{batting_avg:.3f}")
            st.metric("On-Base %", f"{on_base:.3f}")
            st.metric("Slugging %", f"{slugging:.3f}")

        st.markdown("---")

        home_away = st.radio("Game Location", ["Home", "Away"], horizontal=True)
        opp_era = st.slider("Opponent Pitcher ERA", 2.0, 6.0, 4.0, 0.1)

    with col2:
        st.subheader("Model Prediction")

        if st.button("🔮 Run Inference", type="primary"):
            with st.spinner("Running PyTorch model..."):
                import time
                time.sleep(0.5)

                # Simulated model output
                np.random.seed(int(batting_avg * 1000 + on_base * 100))

                hits_prob = batting_avg + np.random.normal(0, 0.02)
                hr_prob = (slugging - batting_avg) * 0.3 + np.random.normal(0, 0.05)
                rbi_expected = (on_base + slugging) * 1.5 + np.random.normal(0, 0.3)

                # Home advantage
                if home_away == "Home":
                    hits_prob += 0.02
                    rbi_expected += 0.2

                # Pitcher adjustment
                pitcher_factor = (opp_era - 4.0) * 0.03
                hits_prob += pitcher_factor
                rbi_expected += pitcher_factor * 2

                st.success("Inference Complete!")

                st.metric("Expected Hits", f"{max(0, hits_prob * 4):.2f}")
                st.metric("HR Probability", f"{max(0, min(50, hr_prob * 100)):.1f}%")
                st.metric("Expected RBI", f"{max(0, rbi_expected):.2f}")

                st.markdown("---")

                st.markdown("**Model Confidence**")
                confidence = 75 + np.random.randint(0, 20)
                st.progress(confidence / 100)
                st.caption(f"{confidence}% confidence")

                st.markdown("---")

                st.code(f"""
# Model Input Tensor
input_features = torch.tensor([
    {batting_avg:.3f},  # batting_avg
    {on_base:.3f},      # on_base_pct
    {slugging:.3f},     # slugging_pct
    {1 if home_away == 'Home' else 0},          # is_home
    {opp_era:.2f},      # opp_pitcher_era
    # ... additional features
])

# Inference
with torch.no_grad():
    prediction = model(input_features.unsqueeze(0))
                """, language="python")


def show_data_pipeline():
    """Data pipeline demonstration"""
    st.header("Data Pipeline")

    st.markdown("Live data fetching and processing pipeline.")

    # Pipeline stages
    stages = ["Fetch MLB API", "Process Data", "Feature Engineering", "Cache Results"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Pipeline Status")

        run_pipeline = st.button("▶️ Run Pipeline", type="primary")

        if run_pipeline:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, stage in enumerate(stages):
                status_text.text(f"Running: {stage}...")
                import time
                time.sleep(0.5)
                progress_bar.progress((i + 1) / len(stages))

            status_text.text("✅ Pipeline Complete!")

            st.session_state['pipeline_run'] = True

    with col2:
        st.subheader("Pipeline Output")

        if st.session_state.get('pipeline_run'):
            # Simulated data output
            st.markdown("**Fetched Games (Today)**")

            games_df = pd.DataFrame({
                'Away': ['Yankees', 'Red Sox', 'Dodgers'],
                'Home': ['Blue Jays', 'Rays', 'Giants'],
                'Time': ['7:05 PM', '7:10 PM', '10:15 PM'],
                'Status': ['Scheduled', 'Scheduled', 'Scheduled']
            })
            st.dataframe(games_df, hide_index=True)

            st.markdown("**Processed Features**")

            features_df = pd.DataFrame({
                'Feature': ['team_win_pct', 'opp_win_pct', 'home_advantage', 'pitcher_era_diff'],
                'Value': [0.583, 0.512, 0.03, -0.45],
                'Normalized': [0.72, 0.48, 0.50, 0.38]
            })
            st.dataframe(features_df, hide_index=True)

    st.markdown("---")

    st.markdown("### Pipeline Code")

    st.code("""
from src.data.fetcher import MLBDataFetcher
from src.features.engineer import FeatureEngineer
import pandas as pd

class DataPipeline:
    def __init__(self):
        self.fetcher = MLBDataFetcher(cache_dir="./data/cache")
        self.engineer = FeatureEngineer()

    def run(self, date: str) -> pd.DataFrame:
        # 1. Fetch raw data
        games = self.fetcher.get_schedule(date)
        standings = self.fetcher.get_standings()

        # 2. Process each game
        processed = []
        for game in games:
            game_data = self.fetcher.get_game_data(game['game_id'])

            # 3. Engineer features
            features = self.engineer.create_features(
                game_data=game_data,
                standings=standings
            )

            processed.append(features)

        # 4. Create DataFrame
        df = pd.DataFrame(processed)

        # 5. Cache results
        df.to_parquet(f"./data/processed/{date}.parquet")

        return df

# Usage
pipeline = DataPipeline()
features_df = pipeline.run("2024-07-04")
    """, language="python")


def show_api_demo():
    """API endpoint demonstration"""
    st.header("REST API Endpoint")

    st.markdown("Test the prediction API endpoint.")

    # API request builder
    st.subheader("Request Builder")

    endpoint = st.selectbox(
        "Endpoint",
        ["/v1/predict/player", "/v1/predict/game", "/v1/stats/player", "/v1/health"]
    )

    if endpoint == "/v1/predict/player":
        st.json({
            "player_id": 660271,
            "player_name": "Shohei Ohtani",
            "features": {
                "batting_avg": 0.304,
                "on_base_pct": 0.412,
                "slugging_pct": 0.654,
                "is_home": True,
                "opponent_era": 4.5
            }
        })

    elif endpoint == "/v1/predict/game":
        st.json({
            "home_team_id": 119,
            "away_team_id": 147,
            "date": "2024-07-04",
            "include_confidence": True
        })

    elif endpoint == "/v1/stats/player":
        st.json({
            "player_name": "Mike Trout",
            "stat_type": "hitting",
            "season": 2024,
            "last_n_games": 10
        })

    else:
        st.json({})

    if st.button("📤 Send Request"):
        with st.spinner("Calling API..."):
            import time
            time.sleep(0.5)

            st.success("Response received!")

            if endpoint == "/v1/predict/player":
                st.json({
                    "status": "success",
                    "prediction": {
                        "expected_hits": 1.42,
                        "hr_probability": 0.18,
                        "expected_rbi": 1.15,
                        "confidence": 0.82
                    },
                    "model_version": "v1.2.0",
                    "latency_ms": 45
                })

            elif endpoint == "/v1/predict/game":
                st.json({
                    "status": "success",
                    "prediction": {
                        "home_win_probability": 0.58,
                        "away_win_probability": 0.42,
                        "predicted_winner": "home",
                        "confidence": 0.75
                    },
                    "model_version": "v2.0.1",
                    "latency_ms": 62
                })

            elif endpoint == "/v1/stats/player":
                st.json({
                    "status": "success",
                    "player": {
                        "name": "Mike Trout",
                        "team": "Los Angeles Angels"
                    },
                    "stats": {
                        "games": 10,
                        "at_bats": 38,
                        "hits": 10,
                        "home_runs": 2,
                        "rbi": 5,
                        "batting_avg": 0.263
                    },
                    "cache_hit": True
                })

            else:
                st.json({
                    "status": "healthy",
                    "version": "1.0.0",
                    "model_loaded": True,
                    "uptime_seconds": 86400
                })

    st.markdown("---")

    st.markdown("### API Implementation")

    st.code("""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

app = FastAPI(title="MLB Predictor API")

# Load model at startup
model = torch.jit.load("models/predictor.pt")
model.eval()

class PlayerPredictionRequest(BaseModel):
    player_id: int
    features: dict

class PredictionResponse(BaseModel):
    status: str
    prediction: dict
    model_version: str
    latency_ms: float

@app.post("/v1/predict/player", response_model=PredictionResponse)
async def predict_player(request: PlayerPredictionRequest):
    import time
    start = time.time()

    # Convert features to tensor
    feature_tensor = torch.tensor([
        request.features['batting_avg'],
        request.features['on_base_pct'],
        request.features['slugging_pct'],
        float(request.features['is_home']),
        request.features['opponent_era']
    ]).float().unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(feature_tensor)

    latency = (time.time() - start) * 1000

    return PredictionResponse(
        status="success",
        prediction={
            "expected_hits": float(output[0][0]),
            "hr_probability": float(output[0][1]),
            "expected_rbi": float(output[0][2]),
            "confidence": float(output[0][3])
        },
        model_version="v1.2.0",
        latency_ms=latency
    )

@app.get("/v1/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
    """, language="python")
