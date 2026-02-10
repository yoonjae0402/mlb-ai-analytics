"""Backend configuration using Pydantic settings."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database — defaults to SQLite for local dev; use PostgreSQL in production
    database_url: str = "sqlite+aiosqlite:///./mlb_analytics.db"
    database_url_sync: str = "sqlite:///./mlb_analytics.db"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    # Model paths
    model_dir: str = "models"
    checkpoint_dir: str = "models/checkpoints"

    # Training defaults
    default_epochs: int = 30
    default_lr: float = 0.001
    default_hidden_size: int = 64
    default_batch_size: int = 32

    # Data
    cache_dir: str = "data/cache"
    data_seasons: list[int] = [2024, 2025]

    # API
    api_prefix: str = "/v1"

    model_config = {"env_prefix": "MLB_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
