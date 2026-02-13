"""Central model orchestration service. Singleton."""

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.predictor import PlayerLSTM
from src.models.xgboost_model import XGBoostPredictor
from src.models.ensemble import EnsemblePredictor
from src.models.model_registry import (
    TrainedModel, compute_metrics, train_lstm, train_xgboost
)
from src.data.feature_builder import (
    build_sequences_from_db, build_player_sequence, temporal_split,
    FEATURE_NAMES, TARGET_NAMES, NUM_FEATURES
)

logger = logging.getLogger(__name__)


class ModelService:
    """Framework-agnostic model service for training, prediction, and ensemble."""

    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.lstm_model: Optional[TrainedModel] = None
        self.xgboost_model: Optional[TrainedModel] = None
        self.ensemble: Optional[EnsemblePredictor] = None

        self.training_status = {
            "is_training": False,
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_model": "",
            "train_loss": 0.0,
            "val_loss": 0.0,
        }

        # Stored training data for attention viz etc.
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None

        self._train_curves: dict = {}

    def train_models(self, db_session, config: dict) -> dict:
        """Train LSTM + XGBoost on real data from DB.

        Args:
            db_session: Sync SQLAlchemy session
            config: Training config dict (epochs, lr, hidden_size, etc.)

        Returns:
            Dict with training results for both models.
        """
        self.training_status["is_training"] = True
        self.training_status["current_model"] = "Loading data"

        try:
            # Load real data
            seasons = config.get("seasons", [2023, 2024])
            X, y, player_ids = build_sequences_from_db(
                db_session, seq_length=10, seasons=seasons
            )
            if len(X) == 0:
                raise ValueError("No training data in database. Run seed_database first.")

            split = temporal_split(X, y, player_ids)
            self.X_train = split["X_train"]
            self.y_train = split["y_train"]
            self.X_val = split["X_val"]
            self.y_val = split["y_val"]

            results = {}

            # Train LSTM
            epochs = config.get("epochs", 30)
            self.training_status["current_model"] = "LSTM"
            self.training_status["total_epochs"] = epochs

            def lstm_progress(epoch, total, train_loss, val_loss):
                self.training_status["current_epoch"] = epoch + 1
                self.training_status["progress"] = (epoch + 1) / total * 50  # 0-50%
                self.training_status["train_loss"] = train_loss
                self.training_status["val_loss"] = val_loss

            self.lstm_model = train_lstm(
                self.X_train, self.y_train, self.X_val, self.y_val,
                epochs=epochs,
                lr=config.get("lr", 0.001),
                hidden_size=config.get("hidden_size", 64),
                batch_size=config.get("batch_size", 32),
                progress_callback=lstm_progress,
            )

            # Save LSTM checkpoint
            version = f"v{int(time.time())}"
            lstm_path = self.checkpoint_dir / f"lstm_{version}.pth"
            self.lstm_model.model.save(lstm_path)
            results["lstm"] = {
                "metrics": self.lstm_model.metrics,
                "train_time": self.lstm_model.train_time,
                "checkpoint": str(lstm_path),
                "version": version,
            }

            # Train XGBoost
            self.training_status["current_model"] = "XGBoost"

            def xgb_progress(step, total, train_loss, val_loss):
                self.training_status["progress"] = 50 + (step + 1) / total * 50  # 50-100%
                self.training_status["train_loss"] = train_loss
                self.training_status["val_loss"] = val_loss

            self.xgboost_model = train_xgboost(
                self.X_train, self.y_train, self.X_val, self.y_val,
                n_estimators=config.get("n_estimators", 200),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("xgb_lr", 0.1),
                progress_callback=xgb_progress,
            )

            # Save XGBoost checkpoint
            xgb_path = self.checkpoint_dir / f"xgboost_{version}.pkl"
            self.xgboost_model.model.save(xgb_path)
            results["xgboost"] = {
                "metrics": self.xgboost_model.metrics,
                "train_time": self.xgboost_model.train_time,
                "checkpoint": str(xgb_path),
                "version": version,
            }

            # Store training curves
            self._train_curves = {
                "lstm": {
                    "train_losses": self.lstm_model.train_losses,
                    "val_losses": self.lstm_model.val_losses,
                },
                "xgboost": {
                    "train_losses": self.xgboost_model.train_losses,
                    "val_losses": self.xgboost_model.val_losses,
                },
            }

            self.training_status["progress"] = 100.0

            return results

        finally:
            self.training_status["is_training"] = False

    def predict_player(
        self, db_session, player_id: int, model_type: str = "lstm"
    ) -> dict:
        """Predict next-game stats for a player.

        Args:
            db_session: Sync SQLAlchemy session
            player_id: Internal player ID
            model_type: "lstm", "xgboost", or "ensemble"

        Returns:
            Dict with predictions and feature values.
        """
        sequence = build_player_sequence(db_session, player_id)
        if sequence is None:
            raise ValueError(f"Insufficient data for player {player_id}")

        predictions = {}
        uncertainty_info = {}

        if model_type in ("lstm", "ensemble") and self.lstm_model:
            # Use MC Dropout for uncertainty estimation
            mc_result = self.lstm_model.model.predict_with_uncertainty(sequence, n_samples=30)
            predictions["lstm"] = mc_result["mean"].flatten().tolist()
            uncertainty_info = {
                "confidence_interval_low": mc_result["ci_low"].flatten().tolist(),
                "confidence_interval_high": mc_result["ci_high"].flatten().tolist(),
                "uncertainty": mc_result["std"].flatten().tolist(),
            }

        if model_type in ("xgboost", "ensemble") and self.xgboost_model:
            xgb_pred = self.xgboost_model.model.predict(sequence)
            predictions["xgboost"] = xgb_pred.flatten().tolist()

        if model_type == "ensemble" and len(predictions) == 2:
            ens = EnsemblePredictor(strategy="weighted_average", weights=[0.5, 0.5])
            ens_pred = ens.predict([
                np.array(predictions["lstm"]).reshape(1, -1),
                np.array(predictions["xgboost"]).reshape(1, -1),
            ])
            predictions["ensemble"] = ens_pred.flatten().tolist()

        # Use the requested model's prediction as primary
        primary = predictions.get(model_type, predictions.get("lstm", [0, 0, 0, 0]))

        result = {
            "player_id": player_id,
            "model_type": model_type,
            "predicted_hits": primary[0],
            "predicted_hr": primary[1],
            "predicted_rbi": primary[2],
            "predicted_walks": primary[3],
            "all_predictions": predictions,
            "feature_names": FEATURE_NAMES,
            "last_features": sequence[0, -1, :].tolist() if sequence is not None else [],
        }

        # Attach uncertainty info if available (from LSTM MC Dropout)
        if uncertainty_info:
            result.update(uncertainty_info)

        return result

    def get_attention_weights(self, sample_idx: int = 0) -> dict:
        """Get attention weights for a validation sample."""
        if self.lstm_model is None or self.X_val is None:
            raise ValueError("Model not trained yet")

        idx = min(sample_idx, len(self.X_val) - 1)
        x = torch.FloatTensor(self.X_val[idx:idx+1])

        self.lstm_model.model.eval()
        with torch.no_grad():
            pred, attn_weights = self.lstm_model.model.forward_with_attention(x)

        return {
            "attention_weights": attn_weights.numpy().tolist(),
            "prediction": pred.numpy().flatten().tolist(),
            "actual": self.y_val[idx].tolist() if self.y_val is not None else [],
            "feature_names": FEATURE_NAMES,
            "target_names": TARGET_NAMES,
            "sample_idx": idx,
            "n_samples": len(self.X_val),
        }

    def get_feature_attribution(self, sample_idx: int = 0) -> dict:
        """Gradient-based feature importance for a sample."""
        if self.lstm_model is None or self.X_val is None:
            raise ValueError("Model not trained yet")

        idx = min(sample_idx, len(self.X_val) - 1)
        x = torch.FloatTensor(self.X_val[idx:idx+1])
        x.requires_grad_(True)

        self.lstm_model.model.eval()
        pred = self.lstm_model.model(x)
        pred.sum().backward()

        grads = x.grad.abs().numpy()[0]  # (seq_length, features)

        # Feature importance = mean absolute gradient across time
        feature_importance = grads.mean(axis=0).tolist()

        return {
            "feature_importance": feature_importance,
            "feature_timestep_grads": grads.tolist(),
            "feature_names": FEATURE_NAMES,
            "sample_idx": idx,
        }

    def get_ensemble_prediction(
        self, db_session, player_id: int, strategy: str = "weighted_average",
        weights: list[float] = None,
    ) -> dict:
        """Ensemble prediction with configurable strategy."""
        if not self.lstm_model or not self.xgboost_model:
            raise ValueError("Both models must be trained first")

        sequence = build_player_sequence(db_session, player_id)
        if sequence is None:
            raise ValueError(f"Insufficient data for player {player_id}")

        lstm_pred = self.lstm_model.model.predict(sequence).reshape(1, -1)
        xgb_pred = self.xgboost_model.model.predict(sequence).reshape(1, -1)

        if weights is None:
            weights = [0.5, 0.5]

        ens = EnsemblePredictor(strategy=strategy, weights=weights)

        if strategy == "stacking" and self.X_val is not None:
            lstm_val = self.lstm_model.model.predict(self.X_val).reshape(-1, 4)
            xgb_val = self.xgboost_model.model.predict(self.X_val).reshape(-1, 4)
            ens.fit_stacking([lstm_val, xgb_val], self.y_val)

        ens_pred = ens.predict([lstm_pred, xgb_pred])

        return {
            "player_id": player_id,
            "strategy": strategy,
            "weights": weights,
            "predictions": {
                "lstm": lstm_pred.flatten().tolist(),
                "xgboost": xgb_pred.flatten().tolist(),
                "ensemble": ens_pred.flatten().tolist(),
            },
            "target_names": TARGET_NAMES,
        }

    def get_weight_sensitivity(self) -> dict:
        """Sweep ensemble weights from 0 to 1 and compute MSE at each point."""
        if not self.lstm_model or not self.xgboost_model or self.X_val is None:
            raise ValueError("Both models must be trained first")

        lstm_pred = self.lstm_model.model.predict(self.X_val).reshape(-1, 4)
        xgb_pred = self.xgboost_model.model.predict(self.X_val).reshape(-1, 4)

        steps = 21
        results = []
        for i in range(steps):
            w = i / (steps - 1)
            ens = EnsemblePredictor(strategy="weighted_average", weights=[w, 1 - w])
            pred = ens.predict([lstm_pred, xgb_pred])
            mse = float(np.mean((pred - self.y_val) ** 2))
            results.append({"lstm_weight": round(w, 2), "mse": mse})

        return {"sweep": results}

    def get_training_status(self) -> dict:
        """Return current training status."""
        return dict(self.training_status)

    def get_training_curves(self) -> dict:
        """Return stored training loss curves."""
        return self._train_curves

    @property
    def is_trained(self) -> bool:
        return self.lstm_model is not None or self.xgboost_model is not None
