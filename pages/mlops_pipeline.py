"""MLOps Pipeline page - MLflow, Docker, Kubernetes showcase"""

import streamlit as st


def show():
    st.title("🚀 MLOps Pipeline")

    st.markdown("""
    Production-grade MLOps practices for model deployment and monitoring.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "MLflow Tracking",
        "Docker",
        "Kubernetes",
        "CI/CD Pipeline"
    ])

    with tab1:
        show_mlflow()

    with tab2:
        show_docker()

    with tab3:
        show_kubernetes()

    with tab4:
        show_cicd()


def show_mlflow():
    """MLflow experiment tracking"""
    st.header("MLflow Experiment Tracking")

    st.markdown("""
    ### Experiment Management

    We use MLflow for:
    - Experiment tracking
    - Model versioning
    - Model registry
    - Artifact storage
    """)

    st.code("""
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlb-player-prediction")

# Start a run
with mlflow.start_run(run_name="lstm_v1"):
    # Log parameters
    mlflow.log_params({
        "model_type": "LSTM",
        "hidden_size": 128,
        "num_layers": 2,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "epochs": 100
    })

    # Train model
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)

    # Log model
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="PlayerPredictor"
    )

    # Log artifacts
    mlflow.log_artifact("training_config.yaml")
    mlflow.log_artifact("model_architecture.png")
    """, language="python")

    st.markdown("---")

    # Model Registry
    st.markdown("### Model Registry")

    st.code("""
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = "runs:/<run_id>/model"
mv = client.create_model_version(
    name="PlayerPredictor",
    source=model_uri,
    run_id="<run_id>"
)

# Transition to production
client.transition_model_version_stage(
    name="PlayerPredictor",
    version=mv.version,
    stage="Production"
)

# Load production model
model = mlflow.pytorch.load_model(
    "models:/PlayerPredictor/Production"
)
    """, language="python")

    # Simulated experiment table
    st.markdown("### Recent Experiments")

    import pandas as pd

    experiments = pd.DataFrame({
        'Run Name': ['lstm_v3', 'lstm_v2', 'lstm_v1', 'transformer_v1'],
        'Model': ['LSTM', 'LSTM', 'LSTM', 'Transformer'],
        'Val Loss': [0.234, 0.256, 0.312, 0.289],
        'MAE': [0.089, 0.095, 0.112, 0.098],
        'Status': ['✅ Production', '📦 Staging', '📁 Archived', '📁 Archived']
    })

    st.dataframe(experiments, hide_index=True, use_container_width=True)


def show_docker():
    """Docker containerization"""
    st.header("Docker Containerization")

    st.markdown("""
    ### Multi-stage Docker Build

    Optimized Docker image for ML model serving.
    """)

    st.code("""
# Dockerfile
# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps \\
    --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s \\
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
    """, language="dockerfile")

    st.markdown("---")

    st.markdown("### Docker Compose for Development")

    st.code("""
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/best_model.pt
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./models:/app/models
    depends_on:
      - mlflow
      - redis
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.0
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://user:pass@db:5432/mlflow
      - ARTIFACT_ROOT=s3://mlflow-artifacts/
    command: mlflow server --host 0.0.0.0

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  grafana-data:
    """, language="yaml")


def show_kubernetes():
    """Kubernetes deployment"""
    st.header("Kubernetes Deployment")

    st.markdown("""
    ### Production Kubernetes Manifests

    Scalable, highly-available model serving infrastructure.
    """)

    st.code("""
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlb-predictor
  labels:
    app: mlb-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlb-predictor
  template:
    metadata:
      labels:
        app: mlb-predictor
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: predictor
        image: gcr.io/project/mlb-predictor:v1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_VERSION
          value: "production"
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: mlflow-config
              key: tracking_uri
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /app/cache
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 1Gi
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
---
apiVersion: v1
kind: Service
metadata:
  name: mlb-predictor-service
spec:
  selector:
    app: mlb-predictor
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mlb-predictor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mlb-predictor
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
    """, language="yaml")

    st.markdown("---")

    st.markdown("### Ingress Configuration")

    st.code("""
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlb-predictor-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.mlb-analytics.com
    secretName: mlb-tls
  rules:
  - host: api.mlb-analytics.com
    http:
      paths:
      - path: /v1/predict
        pathType: Prefix
        backend:
          service:
            name: mlb-predictor-service
            port:
              number: 80
    """, language="yaml")


def show_cicd():
    """CI/CD pipeline"""
    st.header("CI/CD Pipeline")

    st.markdown("""
    ### GitHub Actions Workflow

    Automated testing, building, and deployment.
    """)

    st.code("""
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: gcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3

  train:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Train model
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        python training/train.py --config configs/production.yaml

    - name: Evaluate model
      run: |
        python training/evaluate.py --model-uri $MODEL_URI

  build:
    needs: train
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to GCR
      uses: docker/login-action@v3
      with:
        registry: gcr.io
        username: _json_key
        password: ${{ secrets.GCP_SA_KEY }}

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          gcr.io/${{ secrets.GCP_PROJECT }}/mlb-predictor:${{ github.sha }}
          gcr.io/${{ secrets.GCP_PROJECT }}/mlb-predictor:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4

    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v2
      with:
        project_id: ${{ secrets.GCP_PROJECT }}
        service_account_key: ${{ secrets.GCP_SA_KEY }}

    - name: Get GKE credentials
      run: |
        gcloud container clusters get-credentials mlb-cluster \\
          --zone us-central1-a

    - name: Deploy to GKE
      run: |
        kubectl set image deployment/mlb-predictor \\
          predictor=gcr.io/${{ secrets.GCP_PROJECT }}/mlb-predictor:${{ github.sha }}
        kubectl rollout status deployment/mlb-predictor

    - name: Run smoke tests
      run: |
        python tests/smoke_test.py --endpoint $API_ENDPOINT
    """, language="yaml")

    st.markdown("---")

    st.markdown("### Pipeline Visualization")

    st.markdown("""
    ```
    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │  Test   │───▶│  Train  │───▶│  Build  │───▶│ Deploy  │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    pytest +       MLflow        Docker       Kubernetes
    coverage       Tracking      Registry     Rolling Update
    ```
    """)
