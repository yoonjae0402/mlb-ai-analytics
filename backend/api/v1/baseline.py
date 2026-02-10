"""Baseline predictions API endpoint — generate predictions without training."""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/baseline", tags=["baseline"])


@router.post("/generate")
async def generate_baseline_predictions():
    """Generate baseline predictions for all players using historical averages.
    
    No ML model training required. Uses weighted moving averages of
    recent game stats to project next-game performance.
    """
    from backend.db.session import SyncSessionLocal
    from backend.services.baseline_predictor import generate_all_predictions

    session = SyncSessionLocal()
    try:
        count = generate_all_predictions(session)
        return {
            "status": "success",
            "predictions_generated": count,
            "method": "weighted_historical_average",
            "message": f"Generated {count} baseline predictions from historical stats.",
        }
    except Exception as e:
        logger.error(f"Error generating baseline predictions: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        session.close()
