"""FastAPI application — MLB AI Analytics Platform."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings
from backend.db.session import init_db
from backend.api.v1.router import router as v1_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB pool + tables + auto-generate predictions. Shutdown: cleanup."""
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

# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}

# API routes
app.include_router(v1_router)
