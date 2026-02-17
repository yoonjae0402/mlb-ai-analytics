"""FastAPI application — MLB AI Analytics Platform."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings
from backend.db.session import init_db
from backend.api.v1.router import router as v1_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB pool + tables + auto-generate predictions + auto-reload models."""
    logger.info("Starting MLB AI Analytics API...")
    await init_db()
    logger.info("Database initialized")

    # Auto-generate baseline predictions if none exist
    try:
        from backend.db.session import SyncSessionLocal
        from backend.db.models import Prediction
        session = SyncSessionLocal()
        pred_count = session.query(Prediction).count()
        session.close()

        if pred_count == 0:
            logger.info("No predictions found — generating baseline predictions from historical averages...")
            from backend.services.baseline_predictor import generate_all_predictions
            session = SyncSessionLocal()
            try:
                count = generate_all_predictions(session)
                logger.info(f"Auto-generated {count} baseline predictions on startup.")
            finally:
                session.close()
        else:
            logger.info(f"Found {pred_count} existing predictions — skipping auto-generation.")
    except Exception as e:
        logger.warning(f"Could not auto-generate baseline predictions: {e}")

    # Auto-reload latest trained models from DB checkpoints
    try:
        from backend.db.session import SyncSessionLocal
        from backend.core.model_service import get_model_service
        svc = get_model_service()
        session = SyncSessionLocal()
        try:
            reload_status = svc.auto_reload_latest(session)
            loaded = [k for k, v in reload_status.items() if v == "loaded"]
            if loaded:
                logger.info(f"Auto-reloaded models: {loaded}")
            else:
                logger.info("No trained model checkpoints found — train via /v1/train to get started.")
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Could not auto-reload models: {e}")

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="MLB AI Analytics API",
    description="Real-time MLB player performance prediction using deep learning",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic() - start) * 1000)
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration_ms}ms)"
    )
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    return response


# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# API routes
app.include_router(v1_router)
