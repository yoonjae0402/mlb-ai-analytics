"""
Ensemble methods combining predictions from multiple models (LSTM, XGBoost, LightGBM, Linear).

Strategies:
    - weighted_average: Tunable weight blend of model predictions
    - stacking: Ridge regression meta-learner on model outputs
"""

import numpy as np
from typing import Optional

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None


class EnsemblePredictor:
    """
    Combine predictions from multiple models.

    Args:
        strategy: 'weighted_average' or 'stacking'
        weights: Per-model weights for weighted_average (default: equal weights)
    """

    def __init__(
        self,
        strategy: str = "weighted_average",
        weights: Optional[list[float]] = None,
    ):
        self.strategy = strategy
        self.weights = weights
        self._meta_model = None
        self._fitted = False

    def fit_stacking(
        self,
        model_predictions: list[np.ndarray],
        y_true: np.ndarray,
    ) -> dict:
        """
        Fit the stacking meta-learner.

        Args:
            model_predictions: List of (n_samples, n_targets) arrays.
            y_true: (n_samples, n_targets) ground truth.

        Returns:
            Dict with meta-learner training metrics.
        """
        if Ridge is None:
            raise ImportError("scikit-learn is required for stacking.")

        n_targets = y_true.shape[1]
        stacked = np.hstack(model_predictions)  # (n, n_models * n_targets)

        self._meta_model = []
        metrics = {}
        for t in range(n_targets):
            reg = Ridge(alpha=1.0)
            reg.fit(stacked, y_true[:, t])
            preds = reg.predict(stacked)
            metrics[f"target_{t}_mse"] = float(np.mean((preds - y_true[:, t]) ** 2))
            self._meta_model.append(reg)

        self._fitted = True
        return metrics

    def predict(self, model_predictions: list[np.ndarray]) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            model_predictions: List of (n_samples, n_targets) arrays.

        Returns:
            (n_samples, n_targets) ensemble predictions.
        """
        if self.strategy == "weighted_average":
            return self._weighted_average(model_predictions)
        elif self.strategy == "stacking":
            return self._stacking_predict(model_predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_average(self, preds_list: list[np.ndarray]) -> np.ndarray:
        weights = self.weights
        if weights is None:
            weights = [1.0 / len(preds_list)] * len(preds_list)
        weights = np.array(weights)
        weights = weights / weights.sum()

        result = np.zeros_like(preds_list[0])
        for w, p in zip(weights, preds_list):
            result += w * p
        return result

    def _stacking_predict(self, preds_list: list[np.ndarray]) -> np.ndarray:
        if not self._fitted or self._meta_model is None:
            raise RuntimeError("Stacking model not fitted. Call fit_stacking() first.")

        stacked = np.hstack(preds_list)
        n_targets = len(self._meta_model)
        out = np.column_stack([
            self._meta_model[t].predict(stacked) for t in range(n_targets)
        ])
        return out

    @staticmethod
    def optimize_weights(
        model_predictions: list[np.ndarray],
        y_true: np.ndarray,
        n_steps: int = 10,
    ) -> dict:
        """
        Find optimal weights by grid search over the weight simplex.

        For 2 models: sweeps from [1,0] to [0,1] in n_steps intervals.
        For 3+ models: samples n_steps^(n_models-1) combinations on the simplex.

        Returns:
            Dict with 'weights' (list), 'mse' (float), 'sweep' (list of dicts).
        """
        n_models = len(model_predictions)
        best_weights = [1.0 / n_models] * n_models
        best_mse = float("inf")
        sweep = []

        if n_models == 2:
            for i in range(n_steps + 1):
                w = i / n_steps
                weights = [w, 1 - w]
                pred = sum(wt * p for wt, p in zip(weights, model_predictions))
                mse = float(np.mean((pred - y_true) ** 2))
                sweep.append({"weights": weights, "mse": mse})
                if mse < best_mse:
                    best_mse = mse
                    best_weights = weights

        elif n_models == 3:
            # Grid over 3D simplex
            for i in range(n_steps + 1):
                for j in range(n_steps + 1 - i):
                    k = n_steps - i - j
                    weights = [i / n_steps, j / n_steps, k / n_steps]
                    pred = sum(wt * p for wt, p in zip(weights, model_predictions))
                    mse = float(np.mean((pred - y_true) ** 2))
                    sweep.append({"weights": weights, "mse": mse})
                    if mse < best_mse:
                        best_mse = mse
                        best_weights = weights

        else:
            # For 4+ models, use random search on simplex
            rng = np.random.default_rng(42)
            n_samples = n_steps * 10
            for _ in range(n_samples):
                raw = rng.exponential(1.0, n_models)
                weights = (raw / raw.sum()).tolist()
                pred = sum(wt * p for wt, p in zip(weights, model_predictions))
                mse = float(np.mean((pred - y_true) ** 2))
                sweep.append({"weights": weights, "mse": mse})
                if mse < best_mse:
                    best_mse = mse
                    best_weights = weights

        return {"weights": best_weights, "mse": best_mse, "sweep": sweep}
