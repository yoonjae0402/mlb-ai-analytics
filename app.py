"""
MLB AI Analytics Platform

Demonstrates:
- PyTorch/TensorFlow ML Models
- MLOps Best Practices
- AWS/GCP Cloud Deployment
- LLM Fine-tuning & Integration

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
    .skill-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.2rem;
        display: inline-block;
    }
    .tech-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚾ MLB AI Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🧠 PyTorch Models",
            "🚀 MLOps Pipeline",
            "☁️ Cloud Deployment",
            "🤖 LLM Integration",
            "📊 Live Demo"
        ]
    )

    st.markdown("---")
    st.markdown("### Skills Demonstrated")
    st.markdown("✅ PyTorch/TensorFlow")
    st.markdown("✅ MLOps/Deployment")
    st.markdown("✅ AWS/GCP Cloud")
    st.markdown("✅ LLM Fine-tuning")

# Page routing
if page == "🏠 Overview":
    from pages import overview
    overview.show()
elif page == "🧠 PyTorch Models":
    from pages import pytorch_models
    pytorch_models.show()
elif page == "🚀 MLOps Pipeline":
    from pages import mlops_pipeline
    mlops_pipeline.show()
elif page == "☁️ Cloud Deployment":
    from pages import cloud_deployment
    cloud_deployment.show()
elif page == "🤖 LLM Integration":
    from pages import llm_integration
    llm_integration.show()
elif page == "📊 Live Demo":
    from pages import live_demo
    live_demo.show()
