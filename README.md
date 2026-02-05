# MLB AI Analytics Platform

End-to-end Machine Learning platform for baseball analytics demonstrating production-grade ML engineering skills.

## Skills Demonstrated

| Skill | Technologies |
|-------|-------------|
| **PyTorch / Deep Learning** | Custom LSTM, Transformers, Training pipelines, Mixed precision |
| **MLOps / Deployment** | MLflow, Docker, Kubernetes, CI/CD, Model registry |
| **Cloud (AWS/GCP)** | SageMaker, Lambda, Vertex AI, Terraform IaC |
| **LLM / Fine-tuning** | OpenAI API, RAG, LoRA fine-tuning, LangChain agents |

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/mlb-ai-analytics
cd mlb-ai-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

## Project Structure

```
mlb-ai-analytics/
├── app.py                      # Streamlit application
├── pages/                      # Streamlit pages
│   ├── overview.py             # Project overview
│   ├── pytorch_models.py       # PyTorch model showcase
│   ├── mlops_pipeline.py       # MLOps demonstration
│   ├── cloud_deployment.py     # AWS/GCP integration
│   ├── llm_integration.py      # LLM & fine-tuning
│   └── live_demo.py            # Interactive demos
├── src/
│   ├── data/                   # Data fetching & processing
│   │   └── fetcher.py          # MLB API integration
│   └── models/                 # PyTorch model definitions
│       └── predictor.py        # LSTM player predictor
├── training/
│   ├── train.py                # Training script
│   └── configs/                # Training configurations
│       └── lstm_config.yaml
├── mlops/
│   ├── docker/
│   │   ├── Dockerfile          # Production container
│   │   └── docker-compose.yml  # Local development
│   └── kubernetes/
│       └── deployment.yaml     # K8s manifests
├── cloud/
│   ├── aws/
│   │   └── sagemaker_training.py
│   └── gcp/
│       └── vertex_ai_training.py
├── notebooks/                  # Jupyter experiments
├── experiments/                # MLflow experiments
└── models/                     # Saved models
```

## Features

### 1. PyTorch Deep Learning

- **Custom Architectures**: LSTM with attention, Transformers
- **Training Pipeline**: Mixed precision, gradient clipping, early stopping
- **Model Evaluation**: Cross-validation, metrics tracking

```python
from src.models.predictor import PlayerLSTM

model = PlayerLSTM(
    input_size=15,
    hidden_size=128,
    num_layers=2,
    output_size=4
)
```

### 2. MLOps Pipeline

- **Experiment Tracking**: MLflow integration
- **Model Registry**: Version control for models
- **Containerization**: Docker multi-stage builds
- **Orchestration**: Kubernetes with HPA

```bash
# Run training with MLflow tracking
python training/train.py --config training/configs/lstm_config.yaml

# Start MLflow UI
mlflow ui --port 5000

# Build and run Docker
docker-compose -f mlops/docker/docker-compose.yml up
```

### 3. Cloud Deployment

#### AWS SageMaker

```python
from cloud.aws.sagemaker_training import create_training_job, deploy_model

# Create training job
estimator = create_training_job(role)
estimator.fit({"train": "s3://mlb-data/train/"})

# Deploy to endpoint
predictor = deploy_model(model_data, role, serverless=True)
```

#### GCP Vertex AI

```python
from cloud.gcp.vertex_ai_training import create_custom_training_job

job = create_custom_training_job("mlb-predictor-training")
model = job.run(machine_type="n1-standard-8", accelerator_type="NVIDIA_TESLA_T4")
```

### 4. LLM Integration

- **RAG Pipeline**: ChromaDB + OpenAI for baseball Q&A
- **Fine-tuning**: LoRA/QLoRA for domain adaptation
- **Agents**: LangChain multi-tool agents

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4-turbo"),
    retriever=vectorstore.as_retriever()
)
```

## Training

```bash
# Train model locally
python training/train.py \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0001 \
    --hidden-size 128

# Train with config file
python training/train.py --config training/configs/lstm_config.yaml
```

## Deployment

### Docker

```bash
# Build image
docker build -t mlb-predictor -f mlops/docker/Dockerfile .

# Run container
docker run -p 8000:8000 mlb-predictor
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f mlops/kubernetes/deployment.yaml

# Check status
kubectl get pods -l app=mlb-predictor
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/predict/player` | POST | Predict player performance |
| `/v1/predict/game` | POST | Predict game outcome |
| `/v1/stats/player` | GET | Get player statistics |
| `/health` | GET | Health check |

Example request:
```bash
curl -X POST http://localhost:8000/v1/predict/player \
  -H "Content-Type: application/json" \
  -d '{"player_id": 660271, "features": {"batting_avg": 0.304}}'
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| ML/DL | PyTorch, scikit-learn, XGBoost |
| MLOps | MLflow, Docker, Kubernetes, GitHub Actions |
| Cloud | AWS SageMaker, Lambda, GCP Vertex AI |
| LLM | OpenAI, LangChain, ChromaDB, PEFT |
| Data | MLB Stats API, Pandas, NumPy |
| Web | Streamlit, FastAPI |

## Data Sources

- **MLB Stats API**: Real-time game data, player statistics, standings
- **Statcast**: Advanced batting and pitching metrics (via pybaseball)

## Development

```bash
# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ training/
isort src/ training/

# Type checking
mypy src/
```

## License

MIT License

---

Built with PyTorch, MLflow, AWS/GCP, and LangChain.
