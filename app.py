"""
MLB AI Analytics Platform

Interactive ML platform for baseball analytics with real model training,
attention visualization, ensemble methods, and live predictions.

Run: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="MLB AI Analytics",
    page_icon="⚾",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .tech-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .tech-card h3 { margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("⚾ MLB AI Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Model Comparison",
            "Attention Visualizer",
            "Ensemble Lab",
            "Real-Time Dashboard",
            "Prediction Explorer",
            "Architecture & Docs",
        ]
    )

    st.markdown("---")

    # Training status indicator
    from src.models.model_registry import get_registry
    registry = get_registry(st.session_state)
    if registry.is_trained:
        st.markdown("**Status:** Models trained")
        for name, m in registry.models.items():
            st.caption(f"  {name} — MSE: {m.metrics.get('mse', 0):.4f}")
    else:
        st.markdown("**Status:** No models trained")
        st.caption("Go to Model Comparison to start.")

# Page routing
if page == "Home":
    from pages import overview
    overview.show()
elif page == "Model Comparison":
    from pages import model_comparison
    model_comparison.show()
elif page == "Attention Visualizer":
    from pages import attention_viz
    attention_viz.show()
elif page == "Ensemble Lab":
    from pages import ensemble_lab
    ensemble_lab.show()
elif page == "Real-Time Dashboard":
    from pages import realtime_dashboard
    realtime_dashboard.show()
elif page == "Prediction Explorer":
    from pages import prediction_explorer
    prediction_explorer.show()
elif page == "Architecture & Docs":
    from pages import architecture
    architecture.show()
