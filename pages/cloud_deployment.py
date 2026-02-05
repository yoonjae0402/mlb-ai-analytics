"""Cloud Deployment page - AWS and GCP integration"""

import streamlit as st


def show():
    st.title("☁️ Cloud Deployment")

    st.markdown("""
    Production deployments on AWS and GCP with infrastructure as code.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "AWS SageMaker",
        "AWS Lambda",
        "GCP Vertex AI",
        "Terraform IaC"
    ])

    with tab1:
        show_sagemaker()

    with tab2:
        show_lambda()

    with tab3:
        show_vertex_ai()

    with tab4:
        show_terraform()


def show_sagemaker():
    """AWS SageMaker integration"""
    st.header("AWS SageMaker")

    st.markdown("""
    ### Training on SageMaker

    Scalable model training with managed infrastructure.
    """)

    st.code("""
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

# Initialize session
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define PyTorch estimator
estimator = PyTorch(
    entry_point='train.py',
    source_dir='training/',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU instance
    framework_version='2.1.0',
    py_version='py310',
    hyperparameters={
        'epochs': 100,
        'batch-size': 64,
        'learning-rate': 0.0001
    },
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
        {'Name': 'val:loss', 'Regex': 'Val Loss: ([0-9\\.]+)'},
        {'Name': 'val:mae', 'Regex': 'Val MAE: ([0-9\\.]+)'}
    ]
)

# Hyperparameter tuning
tuner = HyperparameterTuner(
    estimator,
    objective_metric_name='val:loss',
    objective_type='Minimize',
    hyperparameter_ranges={
        'learning-rate': ContinuousParameter(1e-5, 1e-2, scaling_type='Logarithmic'),
        'batch-size': IntegerParameter(32, 128),
        'hidden-size': IntegerParameter(64, 256)
    },
    max_jobs=20,
    max_parallel_jobs=4
)

# Start training
tuner.fit({
    'train': 's3://mlb-data/train/',
    'validation': 's3://mlb-data/validation/'
})
    """, language="python")

    st.markdown("---")

    st.markdown("### SageMaker Endpoint Deployment")

    st.code("""
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig

# Create model
model = PyTorchModel(
    model_data='s3://mlb-models/best-model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='2.1.0',
    py_version='py310'
)

# Option 1: Real-time endpoint
predictor = model.deploy(
    initial_instance_count=2,
    instance_type='ml.g4dn.xlarge',
    endpoint_name='mlb-predictor-prod'
)

# Option 2: Serverless endpoint
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,
    max_concurrency=50
)

serverless_predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name='mlb-predictor-serverless'
)

# Invoke endpoint
response = predictor.predict({
    'player_stats': [[0.275, 0.350, 0.450, ...]],
    'game_context': {'home': True, 'opponent_era': 4.5}
})
    """, language="python")

    # Architecture diagram
    st.markdown("### SageMaker Architecture")

    st.code("""
    ┌──────────────────────────────────────────────────────────┐
    │                    AWS Cloud                              │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
    │  │     S3      │    │  SageMaker  │    │   Lambda    │   │
    │  │  (Data &    │───▶│  Training   │───▶│  (Trigger)  │   │
    │  │   Models)   │    │    Job      │    │             │   │
    │  └─────────────┘    └─────────────┘    └─────────────┘   │
    │         │                  │                  │          │
    │         ▼                  ▼                  ▼          │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
    │  │  SageMaker  │    │   Model     │    │  SageMaker  │   │
    │  │  Feature    │───▶│  Registry   │───▶│  Endpoint   │   │
    │  │   Store     │    │             │    │             │   │
    │  └─────────────┘    └─────────────┘    └─────────────┘   │
    │                                               │          │
    │  ┌─────────────┐    ┌─────────────┐          │          │
    │  │ CloudWatch  │◀───│ API Gateway │◀─────────┘          │
    │  │ (Monitoring)│    │             │                      │
    │  └─────────────┘    └─────────────┘                      │
    └──────────────────────────────────────────────────────────┘
    """, language="text")


def show_lambda():
    """AWS Lambda for serverless inference"""
    st.header("AWS Lambda Serverless Inference")

    st.markdown("""
    ### Lambda Function for Real-time Predictions

    Cost-effective serverless inference for variable workloads.
    """)

    st.code("""
# lambda_function.py
import json
import boto3
import torch
import os
from pathlib import Path

# Initialize S3 client
s3 = boto3.client('s3')

# Global model (loaded once per cold start)
model = None

def load_model():
    global model
    if model is None:
        # Download model from S3
        model_path = '/tmp/model.pt'
        s3.download_file(
            os.environ['MODEL_BUCKET'],
            os.environ['MODEL_KEY'],
            model_path
        )

        # Load model
        model = torch.jit.load(model_path)
        model.eval()

    return model

def lambda_handler(event, context):
    try:
        # Parse input
        body = json.loads(event['body'])
        features = torch.tensor(body['features']).float()

        # Load model (cached after cold start)
        model = load_model()

        # Inference
        with torch.no_grad():
            predictions = model(features.unsqueeze(0))

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'predictions': predictions.squeeze().tolist(),
                'model_version': os.environ.get('MODEL_VERSION', 'unknown')
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
    """, language="python")

    st.markdown("---")

    st.markdown("### Lambda Configuration (SAM Template)")

    st.code("""
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    MemorySize: 3008

Resources:
  PredictorFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: lambda_function.lambda_handler
      Runtime: python3.11
      Architectures:
        - x86_64
      Environment:
        Variables:
          MODEL_BUCKET: !Ref ModelBucket
          MODEL_KEY: models/best_model.pt
          MODEL_VERSION: v1.0.0
      Policies:
        - S3ReadPolicy:
            BucketName: !Ref ModelBucket
      Events:
        Predict:
          Type: Api
          Properties:
            Path: /predict
            Method: post
            RestApiId: !Ref PredictorApi

  PredictorApi:
    Type: AWS::Serverless::Api
    Properties:
      StageName: prod
      Cors:
        AllowOrigin: "'*'"
        AllowHeaders: "'Content-Type'"

  ModelBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: mlb-predictor-models
      VersioningConfiguration:
        Status: Enabled

Outputs:
  ApiEndpoint:
    Description: API Gateway endpoint
    Value: !Sub "https://${PredictorApi}.execute-api.${AWS::Region}.amazonaws.com/prod/"
    """, language="yaml")


def show_vertex_ai():
    """GCP Vertex AI integration"""
    st.header("GCP Vertex AI")

    st.markdown("""
    ### Training on Vertex AI

    Managed ML training with AutoML capabilities.
    """)

    st.code("""
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(
    project='mlb-analytics-project',
    location='us-central1',
    staging_bucket='gs://mlb-staging'
)

# Create custom training job
job = aiplatform.CustomTrainingJob(
    display_name='mlb-player-predictor',
    script_path='training/train.py',
    container_uri='us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest',
    requirements=['mlflow', 'pandas', 'scikit-learn'],
    model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest'
)

# Run training
model = job.run(
    dataset=dataset,
    model_display_name='mlb-predictor-v1',
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    args=[
        '--epochs=100',
        '--batch-size=64',
        '--learning-rate=0.0001'
    ]
)

# Deploy to endpoint
endpoint = model.deploy(
    deployed_model_display_name='mlb-predictor-prod',
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=5,
    traffic_split={'0': 100}
)

# Make prediction
prediction = endpoint.predict(instances=[{
    'player_stats': [0.275, 0.350, 0.450],
    'game_context': {'home': True}
}])
    """, language="python")

    st.markdown("---")

    st.markdown("### Vertex AI Pipeline")

    st.code("""
from kfp import dsl
from kfp.v2 import compiler
from google_cloud_pipeline_components import aiplatform as gcc_aip

@dsl.pipeline(
    name='mlb-training-pipeline',
    description='End-to-end MLB model training pipeline'
)
def training_pipeline(
    project: str,
    location: str,
    training_data_uri: str,
):
    # Data preprocessing
    preprocess_op = gcc_aip.CustomContainerTrainingJobRunOp(
        display_name='preprocess-data',
        container_uri='gcr.io/mlb-project/preprocess:latest',
        model_serving_container_image_uri='',
        project=project,
        location=location,
    )

    # Model training
    train_op = gcc_aip.CustomContainerTrainingJobRunOp(
        display_name='train-model',
        container_uri='gcr.io/mlb-project/train:latest',
        model_serving_container_image_uri='gcr.io/mlb-project/serve:latest',
        project=project,
        location=location,
        machine_type='n1-standard-8',
        accelerator_type='NVIDIA_TESLA_T4',
        accelerator_count=1,
    ).after(preprocess_op)

    # Model evaluation
    eval_op = gcc_aip.ModelEvaluationOp(
        model=train_op.outputs['model'],
        target_column_name='target',
        evaluation_config={'metrics': ['mae', 'rmse']}
    )

    # Conditional deployment
    with dsl.Condition(eval_op.outputs['mae'] < 0.1):
        deploy_op = gcc_aip.ModelDeployOp(
            model=train_op.outputs['model'],
            endpoint=endpoint,
            deployed_model_display_name='mlb-predictor',
            machine_type='n1-standard-4',
            min_replica_count=1,
            max_replica_count=5,
        )

# Compile and run
compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path='pipeline.json'
)
    """, language="python")


def show_terraform():
    """Infrastructure as Code with Terraform"""
    st.header("Terraform Infrastructure as Code")

    st.markdown("""
    ### AWS Infrastructure

    Complete infrastructure definition for ML workloads.
    """)

    st.code("""
# main.tf - AWS Infrastructure

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "mlb-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "mlb-analytics-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.0.0"

  cluster_name    = "mlb-analytics-cluster"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    cpu = {
      min_size     = 2
      max_size     = 10
      desired_size = 3

      instance_types = ["m5.xlarge"]
      capacity_type  = "ON_DEMAND"
    }

    gpu = {
      min_size     = 0
      max_size     = 5
      desired_size = 1

      instance_types = ["g4dn.xlarge"]
      capacity_type  = "SPOT"

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# S3 Buckets
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "mlb-model-artifacts-${var.environment}"

  tags = {
    Environment = var.environment
    Project     = "mlb-analytics"
  }
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# SageMaker
resource "aws_sagemaker_endpoint_configuration" "predictor" {
  name = "mlb-predictor-config"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.predictor.name
    initial_instance_count = 2
    instance_type          = "ml.g4dn.xlarge"

    serverless_config {
      max_concurrency   = 50
      memory_size_in_mb = 4096
    }
  }
}

resource "aws_sagemaker_endpoint" "predictor" {
  name                 = "mlb-predictor-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.predictor.name
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "model_latency" {
  alarm_name          = "mlb-predictor-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "Average"
  threshold           = 1000
  alarm_description   = "Model latency exceeds 1 second"

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.predictor.name
  }
}
    """, language="hcl")

    st.markdown("---")

    st.markdown("### GCP Infrastructure")

    st.code("""
# gcp-main.tf

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# GKE Cluster
resource "google_container_cluster" "mlb_cluster" {
  name     = "mlb-analytics-cluster"
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

resource "google_container_node_pool" "cpu_pool" {
  name       = "cpu-pool"
  cluster    = google_container_cluster.mlb_cluster.name
  location   = var.region
  node_count = 3

  node_config {
    machine_type = "n1-standard-4"
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }
}

resource "google_container_node_pool" "gpu_pool" {
  name       = "gpu-pool"
  cluster    = google_container_cluster.mlb_cluster.name
  location   = var.region
  node_count = 0

  node_config {
    machine_type = "n1-standard-8"

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }

    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  autoscaling {
    min_node_count = 0
    max_node_count = 5
  }
}

# Cloud Storage
resource "google_storage_bucket" "ml_artifacts" {
  name     = "mlb-ml-artifacts-${var.project_id}"
  location = var.region

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# Vertex AI
resource "google_vertex_ai_endpoint" "predictor" {
  name         = "mlb-predictor-endpoint"
  display_name = "MLB Predictor"
  location     = var.region
}
    """, language="hcl")
