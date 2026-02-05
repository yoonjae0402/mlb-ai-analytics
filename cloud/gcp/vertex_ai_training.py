"""
GCP Vertex AI Training Script

Demonstrates:
- Vertex AI custom training jobs
- Model deployment to endpoints
- Pipeline orchestration
"""

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from datetime import datetime


def initialize_vertex_ai(project_id: str, location: str = "us-central1"):
    """Initialize Vertex AI client."""
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{project_id}-staging"
    )


def create_custom_training_job(
    display_name: str,
    container_uri: str = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest"
) -> aiplatform.CustomTrainingJob:
    """Create Vertex AI custom training job."""

    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path="training/train.py",
        container_uri=container_uri,
        requirements=["mlflow", "pandas", "scikit-learn", "pyyaml"],
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest"
    )

    return job


def run_training(
    job: aiplatform.CustomTrainingJob,
    dataset: aiplatform.TabularDataset = None
) -> aiplatform.Model:
    """Run training job and return model."""

    model = job.run(
        dataset=dataset,
        model_display_name="mlb-predictor",
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        args=[
            "--epochs=100",
            "--batch-size=64",
            "--learning-rate=0.0001",
            "--hidden-size=128"
        ],
        environment_variables={
            "MLFLOW_TRACKING_URI": "http://mlflow-server:5000"
        }
    )

    return model


def run_hyperparameter_tuning(
    job: aiplatform.CustomTrainingJob,
    max_trial_count: int = 20
) -> aiplatform.HyperparameterTuningJob:
    """Run hyperparameter tuning job."""

    hpt_job = aiplatform.HyperparameterTuningJob(
        display_name="mlb-predictor-hpt",
        custom_job=job,
        metric_spec={
            "val_loss": "minimize"
        },
        parameter_spec={
            "learning-rate": hpt.DoubleParameterSpec(
                min=1e-5, max=1e-2, scale="log"
            ),
            "batch-size": hpt.DiscreteParameterSpec(
                values=[32, 64, 128], scale="unspecified"
            ),
            "hidden-size": hpt.IntegerParameterSpec(
                min=64, max=256, scale="linear"
            ),
            "num-layers": hpt.IntegerParameterSpec(
                min=1, max=4, scale="linear"
            )
        },
        max_trial_count=max_trial_count,
        parallel_trial_count=4,
        search_algorithm="random"  # or "grid"
    )

    hpt_job.run()
    return hpt_job


def deploy_model(
    model: aiplatform.Model,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    min_replicas: int = 1,
    max_replicas: int = 5
) -> aiplatform.Endpoint:
    """Deploy model to Vertex AI endpoint."""

    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=f"mlb-predictor-endpoint-{datetime.now().strftime('%Y%m%d')}"
    )

    # Deploy model
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="mlb-predictor-deployed",
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=1,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_split={"0": 100},
        service_account="vertex-ai-sa@PROJECT_ID.iam.gserviceaccount.com"
    )

    return endpoint


def create_pipeline():
    """Create Vertex AI Pipeline using KFP."""
    from kfp import dsl
    from kfp.v2 import compiler

    @dsl.component(base_image="python:3.10")
    def preprocess_data(input_path: str, output_path: str):
        """Preprocess training data."""
        import pandas as pd

        df = pd.read_csv(input_path)
        # Preprocessing logic...
        df.to_parquet(output_path)

    @dsl.component(base_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime")
    def train_model(data_path: str, model_path: str):
        """Train PyTorch model."""
        import torch
        # Training logic...

    @dsl.component(base_image="python:3.10")
    def evaluate_model(model_path: str, test_data_path: str) -> float:
        """Evaluate model and return metric."""
        # Evaluation logic...
        return 0.85  # MAE

    @dsl.pipeline(
        name="mlb-training-pipeline",
        description="End-to-end MLB predictor training pipeline"
    )
    def training_pipeline(
        raw_data_path: str,
        model_output_path: str
    ):
        preprocess_task = preprocess_data(
            input_path=raw_data_path,
            output_path="gs://mlb-data/processed/"
        )

        train_task = train_model(
            data_path=preprocess_task.output,
            model_path=model_output_path
        ).after(preprocess_task)

        eval_task = evaluate_model(
            model_path=train_task.output,
            test_data_path="gs://mlb-data/test/"
        ).after(train_task)

    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="pipeline.json"
    )


def main():
    # Initialize
    initialize_vertex_ai(project_id="your-project-id")

    # Create training job
    job = create_custom_training_job(
        display_name="mlb-predictor-training"
    )

    # Run training
    # model = run_training(job)

    # Or run hyperparameter tuning
    # hpt_job = run_hyperparameter_tuning(job)

    # Deploy model
    # endpoint = deploy_model(model)

    print("Vertex AI setup complete!")


if __name__ == "__main__":
    main()
