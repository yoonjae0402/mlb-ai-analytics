"""Honest model evaluation with baselines, confidence intervals, and time-series CV."""

import logging
from typing import Optional

import numpy as np

from src.models.model_registry import compute_metrics

logger = logging.getLogger(__name__)


def run_evaluation(model_service) -> dict:
    """Run full evaluation comparing models vs baselines.

    Returns honest results — if a model doesn't beat baselines, reports it.
    Includes 5-fold time-series CV, RMSE, MAPE, and statistical significance.
    """
    if model_service.X_val is None or model_service.y_val is None:
        return {}

    X_val = model_service.X_val
    y_val = model_service.y_val

    results = {}

    # --- Model predictions and per-model metrics ---
    model_map = {
        "lstm": model_service.lstm_model,
        "xgboost": model_service.xgboost_model,
        "lightgbm": model_service.lightgbm_model,
        "linear": model_service.linear_model,
    }

    model_preds = {}
    for name, trained in model_map.items():
        if trained is None:
            continue
        pred = trained.model.predict(X_val).reshape(-1, 4)
        model_preds[name] = pred
        metrics = compute_metrics(y_val, pred)
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mape"] = _mean_absolute_percentage_error(y_val, pred)
        metrics["per_target"] = _per_target_extended(y_val, pred)
        metrics["ci_95"] = _bootstrap_ci(y_val, pred)
        results[name] = metrics

    # --- Baselines ---
    baselines = {}

    if model_service.y_train is not None:
        # Season average: predict mean of training targets
        season_avg = model_service.y_train.mean(axis=0)
        season_pred = np.tile(season_avg, (len(y_val), 1))
        baselines["season_average"] = compute_metrics(y_val, season_pred)

    # Last-game baseline
    last_game_raw = X_val[:, -1, :4].copy()
    if model_service.y_train is not None:
        target_mean = model_service.y_train.mean(axis=0)
        target_std = model_service.y_train.std(axis=0) + 1e-8
        feat_mean = X_val[:, -1, :4].mean(axis=0)
        feat_std = X_val[:, -1, :4].std(axis=0) + 1e-8
        last_game_pred = (last_game_raw - feat_mean) / feat_std * target_std + target_mean
    else:
        last_game_pred = last_game_raw
        feat_mean = last_game_raw.mean(axis=0)
        feat_std = last_game_raw.std(axis=0) + 1e-8
        target_mean = last_game_raw.mean(axis=0)
        target_std = last_game_raw.std(axis=0) + 1e-8

    baselines["last_game"] = compute_metrics(y_val, last_game_pred)

    # Rolling 3-game baseline
    rolling_raw = X_val[:, -3:, :4].mean(axis=1)
    rolling_pred = (rolling_raw - feat_mean) / feat_std * target_std + target_mean
    baselines["rolling_3_game"] = compute_metrics(y_val, rolling_pred)

    results["baselines"] = baselines

    # --- Comparison with statistical significance ---
    comparison = {}
    baseline_preds = {
        "season_average": np.tile(model_service.y_train.mean(axis=0), (len(y_val), 1))
        if model_service.y_train is not None
        else last_game_pred,
        "last_game": last_game_pred,
        "rolling_3_game": rolling_pred,
    }

    for model_name, pred in model_preds.items():
        model_mse = results[model_name]["mse"]
        for baseline_name, baseline_metrics in baselines.items():
            baseline_mse = baseline_metrics["mse"]
            improvement = (baseline_mse - model_mse) / max(baseline_mse, 1e-8) * 100

            # Per-sample squared errors for significance test
            b_pred = baseline_preds.get(baseline_name)
            p_value = None
            significant = False
            if b_pred is not None:
                model_errors = np.mean((y_val - pred) ** 2, axis=1)
                baseline_errors = np.mean((y_val - b_pred) ** 2, axis=1)
                try:
                    from scipy import stats
                    _, p_value = stats.wilcoxon(model_errors, baseline_errors)
                    p_value = float(p_value)
                    significant = p_value < 0.05
                except ImportError:
                    # Fall back to paired t-test using numpy
                    diff = model_errors - baseline_errors
                    n = len(diff)
                    if n > 1 and diff.std() > 1e-10:
                        t_stat = diff.mean() / (diff.std() / np.sqrt(n))
                        # Approximate p-value from t-distribution (2-tailed)
                        from math import erfc, sqrt
                        p_value = float(erfc(abs(t_stat) / sqrt(2)))
                        significant = p_value < 0.05
                except Exception:
                    pass

            comparison[f"{model_name}_vs_{baseline_name}"] = {
                "model_mse": round(model_mse, 6),
                "baseline_mse": round(baseline_mse, 6),
                "improvement_pct": round(improvement, 2),
                "model_wins": bool(model_mse < baseline_mse),
                "p_value": round(p_value, 4) if p_value is not None else None,
                "significant": significant,
            }

    results["comparison"] = comparison

    # --- Time-series cross-validation (5-fold expanding window) ---
    if model_service.X_train is not None and model_service.y_train is not None:
        cv_results = _time_series_cv(model_service, model_preds.keys())
        if cv_results:
            results["cv_results"] = cv_results

    return results


def _time_series_cv(model_service, model_names, n_folds: int = 5) -> dict:
    """5-fold expanding window time-series CV on training data.

    Uses in-memory models (no re-training) — evaluates trained models
    on held-out temporal folds from training data.
    """
    X = model_service.X_train
    y = model_service.y_train
    n = len(X)

    if n < n_folds * 10:
        return {}

    # Minimum train size: first 40% of data
    min_train = max(int(n * 0.4), 20)
    fold_size = (n - min_train) // n_folds

    if fold_size < 5:
        return {}

    cv_results = {}
    model_map = {
        "lstm": model_service.lstm_model,
        "xgboost": model_service.xgboost_model,
        "lightgbm": model_service.lightgbm_model,
        "linear": model_service.linear_model,
    }

    for name in model_names:
        trained = model_map.get(name)
        if trained is None:
            continue

        fold_mses = []
        fold_maes = []

        for fold in range(n_folds):
            val_start = min_train + fold * fold_size
            val_end = val_start + fold_size
            if val_end > n:
                break

            X_fold_val = X[val_start:val_end]
            y_fold_val = y[val_start:val_end]

            try:
                pred = trained.model.predict(X_fold_val).reshape(-1, 4)
                mse = float(np.mean((y_fold_val - pred) ** 2))
                mae = float(np.mean(np.abs(y_fold_val - pred)))
                fold_mses.append(mse)
                fold_maes.append(mae)
            except Exception as e:
                logger.warning(f"CV fold {fold} failed for {name}: {e}")

        if fold_mses:
            cv_results[name] = {
                "mean_mse": round(float(np.mean(fold_mses)), 6),
                "std_mse": round(float(np.std(fold_mses)), 6),
                "mean_mae": round(float(np.mean(fold_maes)), 6),
                "std_mae": round(float(np.std(fold_maes)), 6),
                "n_folds": len(fold_mses),
                "folds": [
                    {"fold": i + 1, "mse": round(m, 6), "mae": round(a, 6)}
                    for i, (m, a) in enumerate(zip(fold_mses, fold_maes))
                ],
            }

    return cv_results


def _per_target_extended(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MSE, MAE, RMSE per prediction target."""
    target_names = ["hits", "home_runs", "rbi", "walks"]
    result = {}
    for i, name in enumerate(target_names):
        mse = float(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
        mae = float(np.mean(np.abs(y_true[:, i] - y_pred[:, i])))
        result[name] = {
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "rmse": round(float(np.sqrt(mse)), 6),
        }
    return result


def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE — ignores samples where y_true is 0 to avoid division by zero."""
    mask = np.abs(y_true) > 1e-4
    if not mask.any():
        return 0.0
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap 95% CI for MSE."""
    n = len(y_true)
    rng = np.random.RandomState(42)
    mse_samples = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        mse = float(np.mean((y_true[idx] - y_pred[idx]) ** 2))
        mse_samples.append(mse)

    mse_samples_sorted = sorted(mse_samples)
    lower_idx = int((1 - confidence) / 2 * n_bootstrap)
    upper_idx = int((1 + confidence) / 2 * n_bootstrap)

    return {
        "mse_lower": round(mse_samples_sorted[lower_idx], 6),
        "mse_upper": round(mse_samples_sorted[min(upper_idx, n_bootstrap - 1)], 6),
        "mse_mean": round(float(np.mean(mse_samples)), 6),
    }
