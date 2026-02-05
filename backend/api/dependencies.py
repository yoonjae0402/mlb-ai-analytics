"""FastAPI dependency injection."""

from backend.db.session import get_db, SyncSessionLocal
from backend.core.model_service import ModelService

# Singleton model service
_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Get the singleton ModelService instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def get_sync_db():
    """Get a sync database session (for training / data operations)."""
    session = SyncSessionLocal()
    try:
        yield session
    finally:
        session.close()
