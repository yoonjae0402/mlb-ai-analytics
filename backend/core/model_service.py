"""Central model orchestration service. Singleton."""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.predictor import PlayerLSTM
from src.models.xgboost_model import XGBoostPredictor
from src.models.ensemble import EnsemblePredictor
from src.models.model_registry import (
    TrainedModel, compute_metrics, train_lstm, train_xgboost,
    train_lightgbm, train_linear,
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
        self.lightgbm_model: Optional[TrainedModel] = None
        self.linear_model: Optional[TrainedModel] = None
        self.ensemble: Optional[EnsemblePredictor] = None

        self._started_at: float = time.time()
        self._last_retrain_at: Optional[float] = None

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
        """Train models on real data from DB.

        Args:
            db_session: Sync SQLAlchemy session
            config: Training config dict (epochs, lr, hidden_size, etc.)
                    Optional keys: train_lightgbm (bool), train_linear (bool)

        Returns:
            Dict with training results for trained models.
        """
        self.training_status["is_training"] = True
        self.training_status["current_model"] = "Loading data"

        try:
            # Determine which models to train
            do_lightgbm = config.get("train_lightgbm", False)
            do_linear = config.get("train_linear", False)

            # Compute progress slice sizes
            n_models = 2 + (1 if do_lightgbm else 0) + (1 if do_linear else 0)
            slice_pct = 100.0 / n_models
            progress_offset = 0.0

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

            # --- Train LSTM ---
            epochs = config.get("epochs", 30)
            self.training_status["current_model"] = "LSTM"
            self.training_status["total_epochs"] = epochs

            def lstm_progress(epoch, total, train_loss, val_loss):
                self.training_status["current_epoch"] = epoch + 1
                self.training_status["progress"] = progress_offset + (epoch + 1) / total * slice_pct
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

            version = f"v{int(time.time())}"
            lstm_path = self.checkpoint_dir / f"lstm_{version}.pth"
            self.lstm_model.model.save(lstm_path)
            results["lstm"] = {
                "metrics": self.lstm_model.metrics,
                "train_time": self.lstm_model.train_time,
                "checkpoint": str(lstm_path),
                "version": version,
            }
            progress_offset += slice_pct

            # --- Train XGBoost ---
            self.training_status["current_model"] = "XGBoost"

            def xgb_progress(step, total, train_loss, val_loss):
                self.training_status["progress"] = progress_offset + (step + 1) / total * slice_pct
                self.training_status["train_loss"] = train_loss
                self.training_status["val_loss"] = val_loss

            self.xgboost_model = train_xgboost(
                self.X_train, self.y_train, self.X_val, self.y_val,
                n_estimators=config.get("n_estimators", 200),
                max_depth=config.get("max_depth", 6),
                learning_rate=config.get("xgb_lr", 0.1),
                progress_callback=xgb_progress,
            )

            xgb_path = self.checkpoint_dir / f"xgboost_{version}.pkl"
            self.xgboost_model.model.save(xgb_path)
            results["xgboost"] = {
                "metrics": self.xgboost_model.metrics,
                "train_time": self.xgboost_model.train_time,
                "checkpoint": str(xgb_path),
                "version": version,
            }
            progress_offset += slice_pct

            # --- Train LightGBM (optional) ---
            if do_lightgbm:
                self.training_status["current_model"] = "LightGBM"

                def lgb_progress(step, total, train_loss, val_loss):
                    self.training_status["progress"] = progress_offset + (step + 1) / total * slice_pct
                    self.training_status["train_loss"] = train_loss
                    self.training_status["val_loss"] = val_loss

                self.lightgbm_model = train_lightgbm(
                    self.X_train, self.y_train, self.X_val, self.y_val,
                    n_estimators=config.get("n_estimators", 200),
                    max_depth=config.get("max_depth", 6),
                    learning_rate=config.get("xgb_lr", 0.1),
                    num_leaves=config.get("num_leaves", 31),
                    progress_callback=lgb_progress,
                )

                lgb_path = self.checkpoint_dir / f"lightgbm_{version}.pkl"
                self.lightgbm_model.model.save(lgb_path)
                results["lightgbm"] = {
                    "metrics": self.lightgbm_model.metrics,
                    "train_time": self.lightgbm_model.train_time,
                    "checkpoint": str(lgb_path),
                    "version": version,
                }
                progress_offset += slice_pct

            # --- Train Linear (optional) ---
            if do_linear:
                self.training_status["current_model"] = "Linear"

                self.linear_model = train_linear(
                    self.X_train, self.y_train, self.X_val, self.y_val,
                    alpha=config.get("linear_alpha", 1.0),
                    model_type=config.get("linear_model_type", "ridge"),
                )

                lin_path = self.checkpoint_dir / f"linear_{version}.pkl"
                self.linear_model.model.save(lin_path)
                results["linear"] = {
                    "metrics": self.linear_model.metrics,
                    "train_time": self.linear_model.train_time,
                    "checkpoint": str(lin_path),
                    "version": version,
                }

            # Store training curves
            self._train_curves = {}
            for key, model in [
                ("lstm", self.lstm_model),
                ("xgboost", self.xgboost_model),
                ("lightgbm", self.lightgbm_model),
                ("linear", self.linear_model),
            ]:
                if model is not None:
                    self._train_curves[key] = {
                        "train_losses": model.train_losses,
                        "val_losses": model.val_losses,
                    }

            self.training_status["progress"] = 100.0
            self._last_retrain_at = time.time()
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
            model_type: "lstm", "xgboost", "lightgbm", "linear", or "ensemble"

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

        if model_type in ("lightgbm", "ensemble") and self.lightgbm_model:
            lgb_pred = self.lightgbm_model.model.predict(sequence)
            predictions["lightgbm"] = lgb_pred.flatten().tolist()

        if model_type in ("linear", "ensemble") and self.linear_model:
            lin_pred = self.linear_model.model.predict(sequence)
            predictions["linear"] = lin_pred.flatten().tolist()

        # Ensemble: equal-weight blend of all available models
        if model_type == "ensemble" and len(predictions) >= 2:
            preds_arrays = [
                np.array(v).reshape(1, -1) for v in predictions.values()
            ]
            ens = EnsemblePredictor(strategy="weighted_average")
            ens_pred = ens.predict(preds_arrays)
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
        """Ensemble prediction with configurable strategy across all trained models."""
        trained = self._get_trained_tree_models()
        if not self.lstm_model and not trained:
            raise ValueError("At least one model must be trained first")

        sequence = build_player_sequence(db_session, player_id)
        if sequence is None:
            raise ValueError(f"Insufficient data for player {player_id}")

        individual_preds = {}
        pred_arrays = []

        if self.lstm_model:
            p = self.lstm_model.model.predict(sequence).reshape(1, -1)
            individual_preds["lstm"] = p.flatten().tolist()
            pred_arrays.append(p)

        if self.xgboost_model:
            p = self.xgboost_model.model.predict(sequence).reshape(1, -1)
            individual_preds["xgboost"] = p.flatten().tolist()
            pred_arrays.append(p)

        if self.lightgbm_model:
            p = self.lightgbm_model.model.predict(sequence).reshape(1, -1)
            individual_preds["lightgbm"] = p.flatten().tolist()
            pred_arrays.append(p)

        if self.linear_model:
            p = self.linear_model.model.predict(sequence).reshape(1, -1)
            individual_preds["linear"] = p.flatten().tolist()
            pred_arrays.append(p)

        n_models = len(pred_arrays)
        if weights is None:
            weights = [1.0 / n_models] * n_models

        ens = EnsemblePredictor(strategy=strategy, weights=weights)

        if strategy == "stacking" and self.X_val is not None:
            val_preds = []
            if self.lstm_model:
                val_preds.append(self.lstm_model.model.predict(self.X_val).reshape(-1, 4))
            if self.xgboost_model:
                val_preds.append(self.xgboost_model.model.predict(self.X_val).reshape(-1, 4))
            if self.lightgbm_model:
                val_preds.append(self.lightgbm_model.model.predict(self.X_val).reshape(-1, 4))
            if self.linear_model:
                val_preds.append(self.linear_model.model.predict(self.X_val).reshape(-1, 4))
            ens.fit_stacking(val_preds, self.y_val)

        ens_pred = ens.predict(pred_arrays)
        individual_preds["ensemble"] = ens_pred.flatten().tolist()

        return {
            "player_id": player_id,
            "strategy": strategy,
            "weights": weights,
            "predictions": individual_preds,
            "target_names": TARGET_NAMES,
            "n_models": n_models,
        }

    def get_weight_sensitivity(self, model_a: str = "lstm", model_b: str = "xgboost") -> dict:
        """Sweep ensemble weights between two models and compute MSE at each point."""
        models = {
            "lstm": self.lstm_model,
            "xgboost": self.xgboost_model,
            "lightgbm": self.lightgbm_model,
            "linear": self.linear_model,
        }
        m_a = models.get(model_a)
        m_b = models.get(model_b)

        if not m_a or not m_b or self.X_val is None:
            raise ValueError("Both selected models must be trained first")

        pred_a = m_a.model.predict(self.X_val).reshape(-1, 4)
        pred_b = m_b.model.predict(self.X_val).reshape(-1, 4)

        steps = 21
        results = []
        for i in range(steps):
            w = i / (steps - 1)
            ens = EnsemblePredictor(strategy="weighted_average", weights=[w, 1 - w])
            pred = ens.predict([pred_a, pred_b])
            mse = float(np.mean((pred - self.y_val) ** 2))
            results.append({
                "lstm_weight": round(w, 2),
                f"{model_a}_weight": round(w, 2),
                "mse": mse,
            })

        return {
            "sweep": results,
            "model_a": model_a,
            "model_b": model_b,
        }

    def get_trained_model_names(self) -> list[str]:
        """Return names of all currently trained models."""
        names = []
        if self.lstm_model:
            names.append("lstm")
        if self.xgboost_model:
            names.append("xgboost")
        if self.lightgbm_model:
            names.append("lightgbm")
        if self.linear_model:
            names.append("linear")
        return names

    def auto_reload_latest(self, db_session) -> dict:
        """Load the latest saved checkpoint for each model type from disk.

        Called on startup so models survive process restarts.
        Returns a dict of {model_type: "loaded" | "not_found" | "error"}.
        """
        from backend.db.models import ModelVersion
        from sqlalchemy import desc

        status = {}
        loaders = {
            "lstm": self._load_lstm_checkpoint,
            "xgboost": self._load_tree_checkpoint,
            "lightgbm": self._load_tree_checkpoint,
            "linear": self._load_tree_checkpoint,
        }

        for model_type in ["lstm", "xgboost", "lightgbm", "linear"]:
            try:
                row = (
                    db_session.query(ModelVersion)
                    .filter_by(model_type=model_type)
                    .order_by(desc(ModelVersion.trained_at))
                    .first()
                )
                if row is None or not row.checkpoint_path:
                    status[model_type] = "not_found"
                    continue

                path = Path(row.checkpoint_path)
                if not path.exists():
                    status[model_type] = "not_found"
                    continue

                loader = loaders[model_type]
                loader(model_type, path, row)
                status[model_type] = "loaded"
                logger.info(f"Auto-reloaded {model_type} from {path}")

            except Exception as e:
                logger.warning(f"Could not reload {model_type}: {e}")
                status[model_type] = "error"

        return status

    def _load_lstm_checkpoint(self, model_type: str, path: Path, row) -> None:
        from src.models.predictor import PlayerPredictor
        from src.data.feature_builder import NUM_FEATURES
        lstm = PlayerPredictor(input_size=NUM_FEATURES)
        lstm.load(path)
        metrics = {
            "mse": row.val_mse or 0.0,
            "r2": row.val_r2 or 0.0,
        }
        self.lstm_model = TrainedModel(
            model=lstm,
            metrics=metrics,
            train_losses=[],
            val_losses=[],
            train_time=0.0,
        )

    def _load_tree_checkpoint(self, model_type: str, path: Path, row) -> None:
        metrics = {
            "mse": row.val_mse or 0.0,
            "r2": row.val_r2 or 0.0,
        }
        if model_type == "xgboost":
            from src.models.xgboost_model import XGBoostPredictor
            model = XGBoostPredictor.load(path)
            self.xgboost_model = TrainedModel(
                model=model, metrics=metrics,
                train_losses=[], val_losses=[], train_time=0.0,
            )
        elif model_type == "lightgbm":
            from src.models.lightgbm_model import LightGBMPredictor
            model = LightGBMPredictor.load(path)
            self.lightgbm_model = TrainedModel(
                model=model, metrics=metrics,
                train_losses=[], val_losses=[], train_time=0.0,
            )
        elif model_type == "linear":
            from src.models.linear_model import LinearPredictor
            model = LinearPredictor.load(path)
            self.linear_model = TrainedModel(
                model=model, metrics=metrics,
                train_losses=[], val_losses=[], train_time=0.0,
            )

    def get_health(self) -> dict:
        """Return service health info (uptime, model status, last retrain)."""
        import datetime
        uptime_s = time.time() - self._started_at
        return {
            "uptime_seconds": round(uptime_s),
            "trained_models": self.get_trained_model_names(),
            "last_retrain_at": (
                datetime.datetime.fromtimestamp(self._last_retrain_at).isoformat()
                if self._last_retrain_at else None
            ),
            "model_metrics": {
                k: getattr(self, f"{k}_model").metrics
                for k in ["lstm", "xgboost", "lightgbm", "linear"]
                if getattr(self, f"{k}_model") is not None
            },
        }

    def _get_trained_tree_models(self) -> list:
        out = []
        for m in [self.xgboost_model, self.lightgbm_model, self.linear_model]:
            if m is not None:
                out.append(m)
        return out

    def get_training_status(self) -> dict:
        """Return current training status."""
        return dict(self.training_status)

    def get_training_curves(self) -> dict:
        """Return stored training loss curves."""
        return self._train_curves

    @property
    def is_trained(self) -> bool:
        return any(m is not None for m in [
            self.lstm_model, self.xgboost_model, self.lightgbm_model, self.linear_model
        ])
