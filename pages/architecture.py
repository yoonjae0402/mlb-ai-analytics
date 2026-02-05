"""
Architecture & Docs Page

System architecture diagram, tech stack reference,
and collapsible code walkthroughs.
"""

import streamlit as st


def show():
    st.title("Architecture & Docs")
    st.markdown("System design, tech stack, and implementation reference.")

    # --- Architecture Diagram (Mermaid) ---
    st.subheader("System Architecture")

    st.markdown("""
    ```mermaid
    graph TD
        A[MLB Stats API] -->|statsapi| B[Data Fetcher]
        B -->|caching| C[Feature Engineering]
        C --> D[Synthetic Data Generator]
        C --> E[Model Training]

        E --> F[PlayerLSTM]
        E --> G[XGBoost]

        F --> H[Model Registry]
        G --> H

        H --> I[Ensemble Predictor]
        H --> J[Attention Visualizer]
        H --> K[Prediction Explorer]

        I --> L[Streamlit UI]
        J --> L
        K --> L

        M[Real-Time Service] -->|polling| A
        M --> N[Win Probability]
        N --> L
        N --> O[Alert Service]
        O -->|webhook| P[Slack / Discord]

        subgraph Training
            E
            F
            G
        end

        subgraph Inference
            H
            I
            J
            K
        end

        subgraph Real-Time
            M
            N
            O
        end
    ```
    """)

    st.caption("Mermaid diagram — renders in environments with Mermaid support.")

    st.markdown("---")

    # --- Tech Stack ---
    st.subheader("Tech Stack")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**ML / Deep Learning**")
        st.markdown("- PyTorch (LSTM, Attention)")
        st.markdown("- XGBoost")
        st.markdown("- scikit-learn (Ridge)")
        st.markdown("- NumPy / Pandas")

    with col2:
        st.markdown("**Frontend**")
        st.markdown("- Streamlit")
        st.markdown("- Plotly")
        st.markdown("- Interactive widgets")

    with col3:
        st.markdown("**Data**")
        st.markdown("- MLB Stats API")
        st.markdown("- Synthetic generator")
        st.markdown("- Feature engineering")

    with col4:
        st.markdown("**Infrastructure**")
        st.markdown("- Docker / Compose")
        st.markdown("- MLflow")
        st.markdown("- AWS SageMaker")
        st.markdown("- GCP Vertex AI")

    st.markdown("---")

    # --- Model Details ---
    st.subheader("Model Architecture Details")

    with st.expander("PlayerLSTM (Bidirectional LSTM + Multi-Head Attention)"):
        st.markdown("""
        **Architecture:**
        - Input: `(batch, seq_length=10, features=15)` player performance sequences
        - 2-layer Bidirectional LSTM (`hidden_size=64-256`)
        - Multi-head self-attention (8 heads) over LSTM outputs
        - LayerNorm + GELU + Dropout → Linear output layer

        **Key methods:**
        - `forward(x)` — standard prediction
        - `forward_with_attention(x)` — returns `(output, attention_weights)` for visualization
        - `predict(features)` — numpy-in/numpy-out convenience wrapper

        **Training:**
        - Adam optimizer with ReduceLROnPlateau scheduler
        - Gradient clipping (max norm 1.0)
        - Xavier/Orthogonal weight initialization
        """)

    with st.expander("XGBoost Multi-Target Predictor"):
        st.markdown("""
        **Architecture:**
        - One `XGBRegressor` per target variable (4 total)
        - Sequence data flattened to summary statistics: `mean, std, last, trend` per feature
        - Input dimension: `n_features * 4 = 60` flat features

        **Configuration:**
        - 200 estimators, max depth 6, learning rate 0.1 (defaults)
        - Feature importance available via `feature_importances()`
        """)

    with st.expander("Ensemble Predictor"):
        st.markdown("""
        **Strategies:**

        1. **Weighted Average** — Tunable LSTM/XGBoost blend via slider
           - Weight sensitivity curve finds optimal split
           - Typically best at 40-60% LSTM / 40-60% XGBoost

        2. **Stacking (Ridge Meta-Learner)** — Learns optimal combination
           - Ridge regression on concatenated model outputs
           - Fitted on training set, evaluated on validation set
           - Often outperforms manual weight tuning
        """)

    st.markdown("---")

    # --- Data Pipeline ---
    st.subheader("Data Pipeline")

    with st.expander("Synthetic Data Generation"):
        st.markdown("""
        **Player Archetypes:**
        - `power` — High SLG, high barrel rate, elevated K rate (e.g., Aaron Judge)
        - `contact` — High AVG, low K rate, moderate power (e.g., Luis Arraez)
        - `balanced` — Well-rounded stat profile (e.g., Mookie Betts)
        - `speedster` — High sprint speed, gap power (e.g., Ronald Acuna Jr.)

        **Sequence Generation:**
        - 200 players, 10-game sequences, 15 features each
        - Hot/cold streak injection (25% hot, 20% cold)
        - Temporal correlation within sequences
        - Correlated targets based on recent performance
        """)

    with st.expander("Feature Engineering"):
        st.markdown("""
        **15 Input Features:**

        | Feature | Description |
        |---------|-------------|
        | batting_avg | Rolling batting average |
        | on_base_pct | On-base percentage |
        | slugging_pct | Slugging percentage |
        | wOBA | Weighted on-base average |
        | barrel_rate | Barrel percentage |
        | exit_velocity | Average exit velocity (mph) |
        | launch_angle | Average launch angle (degrees) |
        | sprint_speed | Sprint speed (ft/s) |
        | k_rate | Strikeout rate |
        | bb_rate | Walk rate |
        | hard_hit_rate | Hard hit percentage |
        | pull_rate | Pull percentage |
        | park_factor | Park factor adjustment |
        | platoon_advantage | Batter/pitcher handedness |
        | days_rest | Days since last game |
        """)

    st.markdown("---")

    # --- MLOps & Cloud ---
    st.subheader("MLOps & Cloud Deployment")

    with st.expander("Docker Configuration"):
        st.markdown("""
        **Multi-stage Dockerfile:**
        - Base: `python:3.11-slim`
        - Non-root user for security
        - Health checks on API endpoint

        **docker-compose services:**
        - API server (port 8000)
        - Streamlit UI (port 8501)
        - MLflow tracking (port 5000)
        - Redis cache (port 6379)
        - Prometheus metrics (port 9090)
        - Grafana dashboards (port 3000)
        """)

    with st.expander("AWS SageMaker Integration"):
        st.markdown("""
        - Training jobs with spot instances
        - Hyperparameter tuning (Bayesian optimization)
        - Serverless inference endpoints
        - S3 for model artifact storage
        """)

    with st.expander("GCP Vertex AI Integration"):
        st.markdown("""
        - Custom training containers
        - Hyperparameter tuning service
        - Kubeflow Pipelines (KFP) orchestration
        - BigQuery for analytics queries
        """)
