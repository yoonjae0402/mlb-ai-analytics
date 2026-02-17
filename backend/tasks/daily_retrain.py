"""Daily model retraining with champion/challenger pattern.

Trains new models on fresh data. Only promotes the new model if it beats
the current champion's validation MSE. Runs on a schedule via the
daily.yml GitHub Actions workflow or `schedule` library.
"""

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def retrain_if_better(config: dict | None = None) -> dict:
    """Train new models and promote only if they beat the current champion.

    Args:
        config: Optional training config overrides. Uses sensible defaults.

    Returns:
        Dict with promotion decisions per model and final MSEs.
    """
    from backend.db.session import SyncSessionLocal
    from backend.core.model_service import get_model_service
    from backend.db.models import ModelVersion
    from sqlalchemy import desc

    if config is None:
        config = {
            "epochs": 30,
            "lr": 0.001,
            "hidden_size": 64,
            "batch_size": 32,
            "n_estimators": 200,
            "max_depth": 6,
            "xgb_lr": 0.1,
            "seasons": [2023, 2024],
            "train_lightgbm": False,
            "train_linear": False,
        }

    svc = get_model_service()
    session = SyncSessionLocal()
    report = {}

    try:
        logger.info("Daily retrain: loading current champion metrics...")

        # Record champion MSEs before retraining
        champion_mse: dict[str, float] = {}
        for model_type in ["lstm", "xgboost", "lightgbm", "linear"]:
            row = (
                session.query(ModelVersion)
                .filter_by(model_type=model_type)
                .order_by(desc(ModelVersion.trained_at))
                .first()
            )
            if row and row.val_mse is not None:
                champion_mse[model_type] = row.val_mse

        logger.info(f"Champion MSEs: {champion_mse}")

        # Train challenger models in a fresh service instance (don't clobber current)
        from backend.core.model_service import ModelService
        challenger_svc = ModelService(checkpoint_dir="models/challenger")
        challenger_results = challenger_svc.train_models(session, config)

        logger.info(f"Challenger training complete: {list(challenger_results.keys())}")

        # Compare and promote
        promoted_any = False
        for model_type, result in challenger_results.items():
            challenger_val_mse = result.get("metrics", {}).get("mse", float("inf"))
            champion_val_mse = champion_mse.get(model_type, float("inf"))

            improves = challenger_val_mse < champion_val_mse * 0.99  # 1% threshold
            report[model_type] = {
                "champion_mse": round(champion_val_mse, 6) if champion_val_mse != float("inf") else None,
                "challenger_mse": round(challenger_val_mse, 6),
                "promoted": improves,
            }

            if improves:
                logger.info(
                    f"Promoting {model_type}: {challenger_val_mse:.4f} < {champion_val_mse:.4f}"
                )
                # Overwrite live model on the singleton
                challenger_model = getattr(challenger_svc, f"{model_type}_model")
                if challenger_model is not None:
                    setattr(svc, f"{model_type}_model", challenger_model)
                    promoted_any = True
            else:
                logger.info(
                    f"Keeping champion {model_type}: {champion_val_mse:.4f} <= {challenger_val_mse:.4f}"
                )

        if promoted_any:
            svc._last_retrain_at = time.time()
            logger.info("Daily retrain: at least one model promoted.")
        else:
            logger.info("Daily retrain: champion models retained (no improvement).")

        report["retrained_at"] = datetime.utcnow().isoformat()
        return report

    except Exception as e:
        logger.error(f"Daily retrain failed: {e}", exc_info=True)
        return {"error": str(e)}

    finally:
        session.close()
