"""Async SQLAlchemy engine and session factory."""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.config import get_settings
from backend.db.models import Base


settings = get_settings()

# Async engine for FastAPI
async_engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Sync engine for data pipeline / Alembic
sync_engine = create_engine(settings.database_url_sync, echo=False)
SyncSessionLocal = sessionmaker(bind=sync_engine)


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create all tables (for development only; use Alembic in production)."""
    from sqlalchemy import text
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Safely add any columns introduced after the initial DB creation
        col_result = await conn.execute(text("PRAGMA table_info(player_stats)"))
        existing_cols = {row[1] for row in col_result}
        for col_name, ddl in [
            ("innings_pitched", "ALTER TABLE player_stats ADD COLUMN innings_pitched FLOAT"),
            ("earned_runs",     "ALTER TABLE player_stats ADD COLUMN earned_runs SMALLINT"),
        ]:
            if col_name not in existing_cols:
                await conn.execute(text(ddl))


def init_db_sync():
    """Create all tables synchronously (for scripts)."""
    Base.metadata.create_all(bind=sync_engine)
