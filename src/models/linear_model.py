"""
Ridge/Lasso linear regression baseline with the same predict/save/load interface
as XGBoostPredictor.

Serves as the simplest possible baseline to verify that more complex models
actually add value.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.models.xgboost_model import _flatten_sequences

try:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler as SklearnScaler
except ImportError:
    Ridge = None
    Lasso = None
    SklearnScaler = None


class LinearPredictor:
    """
    Multi-output linear regression model wrapping one Ridge or Lasso per target.
    Uses scikit-learn which is already a dependency.
    """

    def __init__(
        self,
        n_targets: int = 4,
        model_type: str = "ridge",  # "ridge" or "lasso"
        alpha: float = 1.0,
    ):
        if Ridge is None:
            raise ImportError("scikit-learn is required: pip install scikit-learn")

        self.n_targets = n_targets
        self.model_type = model_type
        self.alpha = alpha
        self.params = dict(alpha=alpha)

        ModelClass = Ridge if model_type == "ridge" else Lasso
        self.models = [ModelClass(**self.params) for _ in range(n_targets)]
        self._scaler = SklearnScaler()
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
        X_scaled = self._scaler.fit_transform(X_flat)

        metrics = {}
        for t in range(self.n_targets):
            self.models[t].fit(X_scaled, y[:, t])
            preds = self.models[t].predict(X_scaled)
            metrics[f"target_{t}_mse"] = float(np.mean((preds - y[:, t]) ** 2))

        self._fitted = True
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all targets. Returns (n, n_targets)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X_flat = _flatten_sequences(X)
        X_scaled = self._scaler.transform(X_flat)
        preds = np.column_stack([m.predict(X_scaled) for m in self.models])
        # Clip negative predictions to 0 (stats can't be negative)
        return np.clip(preds, 0, None)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "models": self.models,
                    "scaler": self._scaler,
                    "params": self.params,
                    "n_targets": self.n_targets,
                    "model_type": self.model_type,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "LinearPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.models = data["models"]
        obj._scaler = data["scaler"]
        obj.params = data["params"]
        obj.n_targets = data["n_targets"]
        obj.model_type = data["model_type"]
        obj.alpha = data["params"]["alpha"]
        obj._fitted = True
        return obj
