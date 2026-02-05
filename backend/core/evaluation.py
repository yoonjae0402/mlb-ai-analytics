"""Honest model evaluation with baselines and confidence intervals."""

import logging
from typing import Optional

import numpy as np
from src.models.model_registry import compute_metrics

logger = logging.getLogger(__name__)


def run_evaluation(model_service) -> dict:
    """Run full evaluation comparing models vs baselines.

    Returns honest results — if LSTM doesn't beat baselines, reports it.
    """
    if model_service.X_val is None or model_service.y_val is None:
        return {}

    X_val = model_service.X_val
    y_val = model_service.y_val
    target_names = ["hits", "home_runs", "rbi", "walks"]

    results = {}

    # --- Model predictions ---
    if model_service.lstm_model:
        lstm_pred = model_service.lstm_model.model.predict(X_val).reshape(-1, 4)
        lstm_metrics = compute_metrics(y_val, lstm_pred)
        lstm_metrics["ci_95"] = _bootstrap_ci(y_val, lstm_pred)
        results["lstm"] = lstm_metrics

    if model_service.xgboost_model:
        xgb_pred = model_service.xgboost_model.model.predict(X_val).reshape(-1, 4)
        xgb_metrics = compute_metrics(y_val, xgb_pred)
        xgb_metrics["ci_95"] = _bootstrap_ci(y_val, xgb_pred)
        results["xgboost"] = xgb_metrics

    # --- Baselines ---
    baselines = {}

    # 1. Season average baseline: predict mean of training targets
    if model_service.y_train is not None:
        season_avg = model_service.y_train.mean(axis=0)
        season_pred = np.tile(season_avg, (len(y_val), 1))
        baselines["season_average"] = compute_metrics(y_val, season_pred)

    # 2. Last-game baseline: use last timestep features as proxy
    last_game_pred = X_val[:, -1, :4]  # Last timestep, first 4 features (avg, obp, slg, woba)
    # Scale to match target range
    if model_service.y_train is not None:
        target_mean = model_service.y_train.mean(axis=0)
        target_std = model_service.y_train.std(axis=0) + 1e-8
        feat_mean = X_val[:, -1, :4].mean(axis=0)
        feat_std = X_val[:, -1, :4].std(axis=0) + 1e-8
        last_game_pred = (last_game_pred - feat_mean) / feat_std * target_std + target_mean
    baselines["last_game"] = compute_metrics(y_val, last_game_pred)

    # 3. Rolling average baseline: mean of last 3 timesteps
    rolling_features = X_val[:, -3:, :4].mean(axis=1)
    if model_service.y_train is not None:
        rolling_pred = (rolling_features - feat_mean) / feat_std * target_std + target_mean
    else:
        rolling_pred = rolling_features
    baselines["rolling_3_game"] = compute_metrics(y_val, rolling_pred)

    results["baselines"] = baselines

    # --- Comparison ---
    comparison = {}
    for model_name in ["lstm", "xgboost"]:
        if model_name in results:
            model_mse = results[model_name]["mse"]
            for baseline_name, baseline_metrics in baselines.items():
                baseline_mse = baseline_metrics["mse"]
                improvement = (baseline_mse - model_mse) / baseline_mse * 100
                comparison[f"{model_name}_vs_{baseline_name}"] = {
                    "model_mse": model_mse,
                    "baseline_mse": baseline_mse,
                    "improvement_pct": round(improvement, 2),
                    "model_wins": model_mse < baseline_mse,
                }

    results["comparison"] = comparison

    return results


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap 95% confidence intervals for MSE."""
    n = len(y_true)
    rng = np.random.RandomState(42)
    mse_samples = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        mse = float(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        mse_samples.append(mse)

    mse_samples = sorted(mse_samples)
    lower_idx = int((1 - confidence) / 2 * n_bootstrap)
    upper_idx = int((1 + confidence) / 2 * n_bootstrap)

    return {
        "mse_lower": mse_samples[lower_idx],
        "mse_upper": mse_samples[upper_idx - 1],
        "mse_mean": float(np.mean(mse_samples)),
    }
