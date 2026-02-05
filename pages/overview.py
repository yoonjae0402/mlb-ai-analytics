"""Overview page - Project introduction and skills showcase"""

import streamlit as st


def show():
    st.title("⚾ MLB AI Analytics Platform")

    st.markdown("""
    ### End-to-End Machine Learning Project for Baseball Analytics

    This project demonstrates production-grade ML engineering skills through
    a real-world baseball analytics application.
    """)

    st.markdown("---")

    # Skills showcase
    st.header("🎯 Skills Demonstrated")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="tech-card">
        <h3>🧠 PyTorch / Deep Learning</h3>
        <ul>
            <li>Custom neural network architectures</li>
            <li>LSTM for time-series prediction</li>
            <li>Transformer models for sequence data</li>
            <li>Model training & evaluation pipelines</li>
            <li>Hyperparameter optimization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tech-card">
        <h3>☁️ Cloud (AWS/GCP)</h3>
        <ul>
            <li>AWS SageMaker for model training</li>
            <li>S3 for data & model storage</li>
            <li>Lambda for serverless inference</li>
            <li>GCP Vertex AI integration</li>
            <li>BigQuery for analytics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="tech-card">
        <h3>🚀 MLOps / Deployment</h3>
        <ul>
            <li>MLflow for experiment tracking</li>
            <li>Docker containerization</li>
            <li>Kubernetes orchestration</li>
            <li>CI/CD with GitHub Actions</li>
            <li>Model versioning & registry</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tech-card">
        <h3>🤖 LLM / Fine-tuning</h3>
        <ul>
            <li>OpenAI API integration</li>
            <li>RAG (Retrieval Augmented Generation)</li>
            <li>LoRA fine-tuning techniques</li>
            <li>Prompt engineering</li>
            <li>LangChain workflows</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Project architecture
    st.header("🏗️ Project Architecture")

    st.code("""
    mlb-ai-analytics/
    ├── app.py                    # Streamlit application
    ├── src/
    │   ├── data/                 # Data fetching & processing
    │   ├── models/               # PyTorch model definitions
    │   └── features/             # Feature engineering
    ├── training/
    │   ├── train.py              # Training scripts
    │   ├── evaluate.py           # Model evaluation
    │   └── configs/              # Training configurations
    ├── mlops/
    │   ├── docker/               # Dockerfiles
    │   ├── kubernetes/           # K8s manifests
    │   └── terraform/            # Infrastructure as code
    ├── cloud/
    │   ├── aws/                  # AWS SageMaker configs
    │   └── gcp/                  # GCP Vertex AI configs
    ├── notebooks/                # Jupyter experiments
    └── experiments/              # MLflow experiments
    """, language="text")

    st.markdown("---")

    # Tech stack
    st.header("🛠️ Tech Stack")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**ML/DL**")
        st.markdown("- PyTorch")
        st.markdown("- scikit-learn")
        st.markdown("- XGBoost")
        st.markdown("- Transformers")

    with col2:
        st.markdown("**MLOps**")
        st.markdown("- MLflow")
        st.markdown("- Docker")
        st.markdown("- Kubernetes")
        st.markdown("- GitHub Actions")

    with col3:
        st.markdown("**Cloud**")
        st.markdown("- AWS SageMaker")
        st.markdown("- AWS Lambda")
        st.markdown("- GCP Vertex AI")
        st.markdown("- Terraform")

    with col4:
        st.markdown("**LLM**")
        st.markdown("- OpenAI API")
        st.markdown("- LangChain")
        st.markdown("- PEFT/LoRA")
        st.markdown("- ChromaDB")

    st.markdown("---")

    # Quick start
    st.header("🚀 Quick Start")

    st.code("""
    # Clone repository
    git clone https://github.com/yourusername/mlb-ai-analytics
    cd mlb-ai-analytics

    # Setup environment
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    # Run Streamlit app
    streamlit run app.py

    # Train model
    python training/train.py --config configs/lstm_config.yaml

    # Run MLflow UI
    mlflow ui --port 5000
    """, language="bash")
