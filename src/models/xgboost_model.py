"""
XGBoost wrapper with a predict/save/load interface matching the PyTorch models.

Handles sequence data by flattening: each (seq_length, n_features) sample
becomes a single vector of summary statistics (mean, std, last, trend per feature).
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def _flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Convert (n, seq_length, features) → (n, features * 4).

    Per feature: mean, std, last value, linear trend slope.
    """
    if X.ndim == 2:
        return X
    n, seq_len, n_feat = X.shape
    out = np.zeros((n, n_feat * 4), dtype=np.float32)
    t = np.arange(seq_len, dtype=np.float32)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    for i in range(n):
        for f in range(n_feat):
            col = X[i, :, f]
            out[i, f * 4 + 0] = col.mean()
            out[i, f * 4 + 1] = col.std()
            out[i, f * 4 + 2] = col[-1]
            # Linear trend slope
            if t_var > 0:
                out[i, f * 4 + 3] = ((t - t_mean) * (col - col.mean())).sum() / t_var
    return out


class XGBoostPredictor:
    """
    Multi-output XGBoost model wrapping one XGBRegressor per target.
    """

    def __init__(
        self,
        n_targets: int = 4,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        if xgb is None:
            raise ImportError("xgboost is required: pip install xgboost")

        self.n_targets = n_targets
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.models = [
            xgb.XGBRegressor(**self.params) for _ in range(n_targets)
        ]
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Train all per-target models.

        Returns:
            Dict with training metrics (MSE per target).
        """
        X_flat = _flatten_sequences(X)
        metrics = {}

        eval_kwargs = {}
        if eval_set is not None:
            X_val_flat = _flatten_sequences(eval_set[0])

        for t in range(self.n_targets):
            kw = {}
            if eval_set is not None:
                kw["eval_set"] = [(X_val_flat, eval_set[1][:, t])]
            self.models[t].fit(X_flat, y[:, t], verbose=verbose, **kw)
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
            pickle.dump({"models": self.models, "params": self.params, "n_targets": self.n_targets}, f)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostPredictor":
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
