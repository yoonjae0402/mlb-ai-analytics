"""Parquet file caching for pybaseball results.

Avoids re-scraping Baseball Savant / FanGraphs on every run.
TTL: 24h for season stats, 1h for daily schedule, 5min for live games.
"""

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Cache TTL in seconds
TTL = {
    "season_stats": 86400,    # 24 hours
    "game_logs": 86400,       # 24 hours
    "statcast": 86400,        # 24 hours
    "schedule": 3600,         # 1 hour
    "live": 300,              # 5 minutes
}


class CacheManager:
    """Parquet-based cache for DataFrames."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return self.cache_dir / f"{safe}.parquet"

    def _meta_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return self.cache_dir / f"{safe}.meta"

    def get(self, key: str, cache_type: str = "season_stats") -> pd.DataFrame | None:
        """Return cached DataFrame if exists and not expired."""
        path = self._cache_path(key)
        meta = self._meta_path(key)

        if not path.exists() or not meta.exists():
            return None

        try:
            with open(meta) as f:
                ts = float(f.read().strip())
            age = time.time() - ts
            ttl = TTL.get(cache_type, 86400)
            if age > ttl:
                logger.debug(f"Cache expired for {key} (age={age:.0f}s, ttl={ttl}s)")
                return None
            df = pd.read_parquet(path)
            logger.debug(f"Cache hit for {key} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache {key}: {e}")
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        """Store DataFrame to parquet cache."""
        if df is None or df.empty:
            return
        path = self._cache_path(key)
        meta = self._meta_path(key)
        try:
            df.to_parquet(path, index=False)
            with open(meta, "w") as f:
                f.write(str(time.time()))
            logger.debug(f"Cached {key} ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Failed to write cache {key}: {e}")

    def invalidate(self, key: str) -> None:
        """Remove a cache entry."""
        for p in [self._cache_path(key), self._meta_path(key)]:
            if p.exists():
                p.unlink()
