#!/bin/bash
set -e

echo "=== MLB AI Analytics Backend ==="

# Auto-seed database if MLB_AUTO_SEED is set and DB is empty
if [ "${MLB_AUTO_SEED}" = "true" ]; then
    echo "Checking if database needs seeding..."
    
    python3 -c "
import logging
logging.basicConfig(level=logging.INFO)

from backend.db.session import SyncSessionLocal, init_db_sync
from backend.db.models import Player

# Create tables
init_db_sync()

session = SyncSessionLocal()
player_count = session.query(Player).count()
session.close()

if player_count == 0:
    print('DATABASE EMPTY — seeding 2025 MLB + AAA data...')
    from src.data.pipeline import MLBDataPipeline
    pipeline = MLBDataPipeline()
    pipeline.seed_database([2025])
    
    print('Generating baseline predictions...')
    from backend.services.baseline_predictor import generate_all_predictions
    session = SyncSessionLocal()
    count = generate_all_predictions(session)
    session.close()
    print(f'Generated {count} baseline predictions.')
else:
    print(f'Database already has {player_count} players — skipping seed.')
"
    echo "Database ready."
fi

# Start the API server
echo "Starting uvicorn..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
