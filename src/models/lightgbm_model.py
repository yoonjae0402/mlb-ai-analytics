"""
LightGBM wrapper with the same predict/save/load interface as XGBoostPredictor.

Uses identical sequence flattening: each (seq_length, n_features) sample
becomes mean/std/last/trend per feature.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.models.xgboost_model import _flatten_sequences

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class LightGBMPredictor:
    """
    Multi-output LightGBM model wrapping one LGBMRegressor per target.
    """

    def __init__(
        self,
        n_targets: int = 4,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        if lgb is None:
            raise ImportError("lightgbm is required: pip install lightgbm")

        self.n_targets = n_targets
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            verbose=-1,
        )
        self.models = [
            lgb.LGBMRegressor(**self.params) for _ in range(n_targets)
        ]
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> dict:
        """
        Train all per-target models.

        Returns:
            Dict with training metrics (MSE per target).
        """
        X_flat = _flatten_sequences(X)
        metrics = {}

        for t in range(self.n_targets):
            kw = {}
            if eval_set is not None:
                X_val_flat = _flatten_sequences(eval_set[0])
                kw["eval_set"] = [(X_val_flat, eval_set[1][:, t])]
            self.models[t].fit(X_flat, y[:, t], **kw)
            preds = self.models[t].predict(X_flat)
            metrics[f"target_{t}_mse"] = float(np.mean((preds - y[:, t]) ** 2))

        self._fitted = True
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all targets. Returns (n, n_targets)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X_flat = _flatten_sequences(X)
        preds = np.column_stack([m.predict(X_flat) for m in self.models])
        return preds

    def feature_importances(self) -> np.ndarray:
        """Get mean feature importance across targets. Shape (n_flat_features,)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        imps = np.array([m.feature_importances_ for m in self.models])
        return imps.mean(axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"models": self.models, "params": self.params, "n_targets": self.n_targets}, f
            )

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.models = data["models"]
        obj.params = data["params"]
        obj.n_targets = data["n_targets"]
        obj._fitted = True
        return obj
