# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLB Baseball Analytics Platform — a FanGraphs-inspired baseball analytics site built with Next.js 14, FastAPI, PostgreSQL, PyTorch, and XGBoost. Uses real MLB data from pybaseball and MLB Stats API. Features a beginner-friendly design with stat tooltips, context badges, and a "Beginner Mode" toggle.

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

## Design System (v3.0 - FanGraphs-Inspired Redesign)

### Color Palette
All colors defined as CSS variables in `frontend/app/globals.css` and Tailwind `fg-*` classes in `tailwind.config.ts`:
- `--color-primary: #5efc8d`   — bright green, active states, highlights
- `--color-secondary: #8ef9f3` — cyan, accents, links, hover
- `--color-accent: #93bedf`    — steel blue, borders, secondary elements
- `--color-panel: #8377d1`     — purple, nav bar, panels, section headers
- `--color-background: #6d5a72`— dark purple-gray, main background
- `--color-card: #7a6580`      — card surfaces
- `--color-text: #f0f4f8`      — primary text
- `--color-muted: #c8d8e8`     — secondary text
- `--color-subtle: #a0b4c8`    — tertiary text

### Navigation
- **Top navigation** (not sidebar) — `frontend/components/layout/TopNav.tsx`
- FanGraphs-style tabs: Games | Schedule | Players | Pitchers | Compare | Leaderboard | Analysis▾
- Sticky header with logo, tabs, search bar, Beginner Mode toggle, API status
- Layout file: `frontend/app/layout.tsx` — uses TopNav + Footer (NO sidebar)

### Beginner Mode
- Toggle in TopNav sets `document.body.classList.add("beginner-mode")`
- CSS hides `.advanced-stat` elements and shows `.beginner-label` elements when active

### Data Tables (FanGraphs-style)
- Use `<table className="fg-table">` for sortable data tables
- Use `<abbr className="stat-abbr" data-tip="...">` for stat header tooltips
- Use `[data-tooltip="..."]` for hover tooltips on any element

## Architecture

### Frontend (`frontend/`)
- Next.js 14 (App Router), TypeScript, Tailwind CSS, Recharts, React Query
- Pages: Home (`/`), Games (`/dashboard`), Schedule, Players, Pitchers, Compare, Leaderboard, Game Detail, Player Detail, Model Comparison, Attention, Ensemble, Tuning, System, Architecture
- **NO prediction model UI visible to users** — prediction pages removed from navigation
- Win probability is framed as "Statistical Analysis" (Pythagorean expectation), NOT "AI prediction"
- API client in `frontend/lib/api.ts`
- Footer: `frontend/components/layout/Footer.tsx`

### Backend (`backend/`)
- **Entry point**: `backend/main.py` — FastAPI app with CORS, lifespan
- **API routes**: `backend/api/v1/` — train, predict, players, attention, ensemble, games, evaluation, data, tuning
- **Model service**: `backend/core/model_service.py` — singleton for training, prediction, ensemble
- **Database**: `backend/db/models.py` — SQLAlchemy ORM (players, player_stats, games, predictions, model_versions)
- **Session**: `backend/db/session.py` — async (asyncpg) + sync (psycopg2) engines

### Data Pipeline (`src/data/`)
- `pipeline.py` — MLBDataPipeline: fetches from pybaseball + statsapi, seeds PostgreSQL
- `feature_builder.py` — builds (N, 10, 26) sequences from player_stats rows
- `cache_manager.py` — Parquet file caching with TTL
- `fetcher.py` — MLB Stats API integration with JSON caching + retry

### Models (`src/models/`)
- `predictor.py` — PlayerLSTM (BiLSTM + attention), PlayerPredictor
- `xgboost_model.py` — XGBoostPredictor with sequence flattening
- `lightgbm_model.py` — LightGBMPredictor
- `linear_model.py` — LinearPredictor (Ridge/Lasso)
- `ensemble.py` — EnsemblePredictor (weighted_average, stacking)
- `model_registry.py` — Training orchestration, compute_metrics, load_training_data

### Key Patterns
- `compute_metrics()` (public) in model_registry computes MSE, MAE, R², per-target
- `build_sequences_from_db()` in feature_builder creates sliding window sequences
- `temporal_split()` prevents data leakage in time-series
- ModelService is a singleton — accessed via `get_model_service()`
- Training runs in FastAPI BackgroundTasks
- Win probability uses Pythagorean expectation (exponent 1.83) in `backend/services/game_predictor.py`

## Environment Notes
- Use `python3` not `python` on this machine
- PostgreSQL: `mlb:mlb@localhost:5432/mlb_analytics`
- Frontend: `http://localhost:3000`, Backend: `http://localhost:8000`
- Node.js 18.2.0 on this machine (works in Docker/CI)
- Python 3.13 on macOS ARM

## Known Bugs Fixed (2026-02-19)
1. Pitcher stats API — fixed nested stats structure in `backend/api/v1/pitchers.py`
2. TrainedModel missing name — fixed `_load_lstm_checkpoint` and `_load_tree_checkpoint` in `model_service.py`
3. Match prediction skew — `_estimate_runs_from_predictions` now fills missing slots with league_avg_per_slot
4. Pitcher lookup bug — `_stats_to_dataframe` now uses O(1) lookup by (opponent, date)
5. Missing OBP/SLG in season_totals — fixed both `/players/compare` and `/players/{id}` endpoints
6. Player detail page wrong field — fixed `season_totals.batting_avg` access
7. Architecture page outdated counts — fixed feature and model counts

## Design Decisions (2026-02-23 Redesign)
- Switched from sidebar navigation to FanGraphs-style top tabs
- Removed prediction model UI from user-facing navigation; model pages accessible via "Analysis" dropdown
- Win probability framed as "Statistical Analysis" with methodology disclaimer
- CSS variables defined in `globals.css` as the single source of truth for colors
- All `mlb-*` Tailwind classes remapped to new purple/green color palette for backward compatibility
- Beginner Mode toggle hides advanced stats and shows plain-English labels site-wide
- Footer shows data sources, last refresh time, version
- Prediction Methodology fully documented in README.md
