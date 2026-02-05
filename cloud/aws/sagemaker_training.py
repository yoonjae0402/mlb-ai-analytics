"""
AWS SageMaker Training Script

Demonstrates:
- SageMaker PyTorch training jobs
- Hyperparameter tuning
- Model deployment to endpoint
"""

import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter
)
from sagemaker.serverless import ServerlessInferenceConfig
import boto3
from datetime import datetime


def create_training_job(
    role: str,
    instance_type: str = "ml.p3.2xlarge",
    instance_count: int = 1
) -> PyTorch:
    """Create SageMaker PyTorch training job."""

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training/",
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        framework_version="2.1.0",
        py_version="py310",
        hyperparameters={
            "epochs": 100,
            "batch-size": 64,
            "learning-rate": 0.0001,
            "hidden-size": 128,
            "num-layers": 2
        },
        metric_definitions=[
            {"Name": "train:loss", "Regex": "Train Loss: ([0-9\\.]+)"},
            {"Name": "val:loss", "Regex": "Val Loss: ([0-9\\.]+)"},
        ],
        enable_sagemaker_metrics=True,
        max_run=3600 * 4,  # 4 hours max
        output_path="s3://mlb-models/training-output/"
    )

    return estimator


def run_hyperparameter_tuning(estimator: PyTorch, max_jobs: int = 20) -> HyperparameterTuner:
    """Run hyperparameter optimization."""

    hyperparameter_ranges = {
        "learning-rate": ContinuousParameter(1e-5, 1e-2, scaling_type="Logarithmic"),
        "batch-size": CategoricalParameter([32, 64, 128]),
        "hidden-size": IntegerParameter(64, 256),
        "num-layers": IntegerParameter(1, 4),
        "dropout": ContinuousParameter(0.1, 0.5)
    }

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name="val:loss",
        objective_type="Minimize",
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=4,
        strategy="Bayesian"
    )

    return tuner


def deploy_model(
    model_data: str,
    role: str,
    instance_type: str = "ml.g4dn.xlarge",
    serverless: bool = False
) -> sagemaker.Predictor:
    """Deploy trained model to SageMaker endpoint."""

    model = PyTorchModel(
        model_data=model_data,
        role=role,
        entry_point="inference.py",
        source_dir="src/",
        framework_version="2.1.0",
        py_version="py310"
    )

    if serverless:
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=4096,
            max_concurrency=50
        )
        predictor = model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=f"mlb-predictor-serverless-{datetime.now().strftime('%Y%m%d')}"
        )
    else:
        predictor = model.deploy(
            initial_instance_count=2,
            instance_type=instance_type,
            endpoint_name=f"mlb-predictor-{datetime.now().strftime('%Y%m%d')}"
        )

    return predictor


def main():
    # Initialize SageMaker session
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    print(f"SageMaker role: {role}")
    print(f"Default bucket: {session.default_bucket()}")

    # Create estimator
    estimator = create_training_job(role)

    # Option 1: Single training job
    # estimator.fit({
    #     "train": "s3://mlb-data/train/",
    #     "validation": "s3://mlb-data/validation/"
    # })

    # Option 2: Hyperparameter tuning
    # tuner = run_hyperparameter_tuning(estimator)
    # tuner.fit({
    #     "train": "s3://mlb-data/train/",
    #     "validation": "s3://mlb-data/validation/"
    # })
    # best_training_job = tuner.best_training_job()

    # Deploy model
    # predictor = deploy_model(
    #     model_data="s3://mlb-models/best-model/model.tar.gz",
    #     role=role,
    #     serverless=True
    # )

    print("SageMaker setup complete!")


if __name__ == "__main__":
    main()
