# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLB AI Analytics Platform — an end-to-end ML platform for baseball analytics built with Next.js 14, FastAPI, PostgreSQL, PyTorch, and XGBoost. Uses real MLB data from pybaseball and MLB Stats API.

## Common Commands

```bash
# Start full stack (Docker)
docker-compose up

# Backend only
uvicorn backend.main:app --reload --port 8000

# Frontend only
cd frontend && npm run dev

# Seed database with real MLB data
python3 -c "from src.data.pipeline import MLBDataPipeline; MLBDataPipeline().seed_database([2024])"

# Run data refresh
python3 -c "from backend.tasks.data_refresh import refresh_daily_data; refresh_daily_data()"
```

## Architecture

### Frontend (`frontend/`)
- Next.js 14 (App Router), TypeScript, Tailwind CSS, Recharts, React Query
- 7 pages: Home, Models, Attention, Ensemble, Dashboard, Predict, Architecture
- Dark MLB theme (#0a1628 background, #e63946 red accent)
- API client in `frontend/lib/api.ts`

### Backend (`backend/`)
- **Entry point**: `backend/main.py` — FastAPI app with CORS, lifespan
- **API routes**: `backend/api/v1/` — train, predict, players, attention, ensemble, games, evaluation, data, tuning
- **Model service**: `backend/core/model_service.py` — singleton for training, prediction, ensemble
- **Database**: `backend/db/models.py` — SQLAlchemy ORM (players, player_stats, games, predictions, model_versions)
- **Session**: `backend/db/session.py` — async (asyncpg) + sync (psycopg2) engines

### Data Pipeline (`src/data/`)
- `pipeline.py` — MLBDataPipeline: fetches from pybaseball + statsapi, seeds PostgreSQL
- `feature_builder.py` — builds (N, 10, 15) sequences from player_stats rows
- `cache_manager.py` — Parquet file caching with TTL
- `fetcher.py` — MLB Stats API integration with JSON caching + retry

### Models (`src/models/`)
- `predictor.py` — PlayerLSTM (BiLSTM + attention), PlayerPredictor, GameTransformer
- `xgboost_model.py` — XGBoostPredictor with sequence flattening
- `ensemble.py` — EnsemblePredictor (weighted_average, stacking)
- `model_registry.py` — Training orchestration, compute_metrics, load_training_data

### Key Patterns
- `compute_metrics()` (public) in model_registry computes MSE, MAE, R², per-target
- `build_sequences_from_db()` in feature_builder creates sliding window sequences
- `temporal_split()` prevents data leakage in time-series
- ModelService is a singleton — accessed via `get_model_service()`
- Training runs in FastAPI BackgroundTasks

## Environment Notes
- Use `python3` not `python` on this machine
- PostgreSQL: `mlb:mlb@localhost:5432/mlb_analytics`
- Frontend: `http://localhost:3000`, Backend: `http://localhost:8000`
