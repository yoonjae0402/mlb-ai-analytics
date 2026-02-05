"""
Model Training Script

Usage:
    python training/train.py --config configs/lstm_config.yaml
    python training/train.py --epochs 100 --lr 0.0001
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import mlflow
import mlflow.pytorch
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predictor import PlayerLSTM


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_synthetic_data(n_samples: int = 10000, seq_length: int = 10, n_features: int = 15):
    """Create synthetic training data for demonstration."""
    np.random.seed(42)

    X = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
    # Target: hits, hr_prob, rbi, confidence
    y = np.random.randn(n_samples, 4).astype(np.float32)
    y = np.abs(y)  # Make positive

    return X, y


def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("mlb-player-prediction")

    with mlflow.start_run(run_name=f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "model_type": "PlayerLSTM",
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dropout": args.dropout
        })

        # Create data
        print("Creating synthetic training data...")
        X, y = create_synthetic_data()

        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # DataLoaders
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # Model
        model = PlayerLSTM(
            input_size=15,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=4,
            dropout=args.dropout
        ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.MSELoss()
        scaler = GradScaler()

        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step()

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)

            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, 'models/best_model.pt')

            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")

        # Log final metrics
        mlflow.log_metric("best_val_loss", best_val_loss)

        print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLB player predictor")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            setattr(args, key.replace("-", "_"), value)

    main(args)
