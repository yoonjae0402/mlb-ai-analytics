"""Optuna hyperparameter tuning for LSTM and XGBoost."""

import logging
from typing import Optional

import numpy as np
import optuna

from src.models.model_registry import train_lstm, train_xgboost

logger = logging.getLogger(__name__)

# Module-level tuning state
_tuning_status = {
    "is_tuning": False,
    "model_type": "",
    "n_trials": 0,
    "completed_trials": 0,
    "best_params": None,
    "best_score": None,
}


def get_tuning_status() -> dict:
    return dict(_tuning_status)


def tune_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Bayesian optimization over LSTM hyperparameters."""
    _tuning_status["is_tuning"] = True
    _tuning_status["model_type"] = "lstm"
    _tuning_status["n_trials"] = n_trials
    _tuning_status["completed_trials"] = 0

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = 15  # Quick training for HPT

        result = train_lstm(
            X_train, y_train, X_val, y_val,
            epochs=epochs, lr=lr, hidden_size=hidden_size, batch_size=batch_size,
        )

        _tuning_status["completed_trials"] = trial.number + 1
        return result.metrics["mse"]

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        _tuning_status["best_params"] = study.best_params
        _tuning_status["best_score"] = study.best_value

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": n_trials,
        }
    finally:
        _tuning_status["is_tuning"] = False


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Bayesian optimization over XGBoost hyperparameters."""
    _tuning_status["is_tuning"] = True
    _tuning_status["model_type"] = "xgboost"
    _tuning_status["n_trials"] = n_trials
    _tuning_status["completed_trials"] = 0

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)

        result = train_xgboost(
            X_train, y_train, X_val, y_val,
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        )

        _tuning_status["completed_trials"] = trial.number + 1
        return result.metrics["mse"]

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        _tuning_status["best_params"] = study.best_params
        _tuning_status["best_score"] = study.best_value

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": n_trials,
        }
    finally:
        _tuning_status["is_tuning"] = False
