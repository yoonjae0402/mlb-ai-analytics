# MLB AI Analytics Platform

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688)

An end-to-end ML platform that predicts MLB player performance using deep learning on real Statcast data. Features a beginner-friendly Next.js dashboard with contextual stat badges, interactive game predictions, and player comparisons.

## What This Does

Predicts next-game batting stats (hits, HR, RBI, walks) for MLB players using a Bidirectional LSTM with attention trained on real game-by-game data from pybaseball and MLB Stats API. Includes model comparison against XGBoost baselines, ensemble methods, and honest evaluation with confidence intervals.

## Architecture

```
pybaseball / MLB Stats API
        |
   PostgreSQL (player_stats, games, predictions)
        |
   Feature Engineering (rolling averages, trends, Statcast metrics)
        |
   Models: BiLSTM + Attention | XGBoost | Ensemble
        |
   FastAPI REST API (async, background training)
        |
   Next.js 14 Dashboard (React Query, Recharts, Tailwind)
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Recharts, React Query |
| **Backend** | FastAPI, SQLAlchemy (async), Pydantic, Uvicorn |
| **Database** | PostgreSQL |
| **ML** | PyTorch (BiLSTM + multi-head attention), XGBoost, Optuna |
| **Data** | pybaseball (Statcast, FanGraphs), MLB Stats API |
| **Deploy** | Docker Compose, Railway |

## Getting Started

### Docker (recommended)

```bash
docker-compose up
```

This starts PostgreSQL, the FastAPI backend (port 8000), and the Next.js frontend (port 3000).

### Manual Setup

```bash
# Backend
pip install -r backend/requirements.txt

# Start PostgreSQL (ensure it's running on port 5432)
# Create database: createdb mlb_analytics

# Seed database with real MLB data
python3 -c "from src.data.pipeline import MLBDataPipeline; MLBDataPipeline().seed_database([2024])"

# Start backend
uvicorn backend.main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Verify

```bash
curl localhost:8000/health              # {"status": "ok", "version": "2.0.0"}
curl "localhost:8000/v1/players/search?q=judge"  # Aaron Judge
```

## Key Features

### 10 Interactive Pages

1. **Dashboard** — Live scores, system status, top predictions
2. **Schedule Calendar** — Week/month view with clickable games
3. **Game Predictions** — Click any game to see AI predictions for every player on both rosters
4. **Player Predict** — Search any player, predict next-game stats with context badges and percentile bars
5. **Player Index** — Browse all players with filtering
6. **Compare Players** — Side-by-side stat comparison with context badges, trend indicators, and winner highlighting
7. **Model Comparison** — Train LSTM & XGBoost side-by-side, live training curves
8. **Attention Visualizer** — Inspect what the LSTM model focuses on (heatmaps + gradient attribution)
9. **Ensemble Lab** — Weighted average vs stacking, weight sensitivity analysis
10. **Architecture & Docs** — System design, API reference, model documentation

### Beginner-Friendly Design System

- **Context Badges** — "Elite" (gold), "Great" (green), "Average" (gray), "Below Avg" (orange) pills next to every stat
- **Percentile Bars** — 0-99 ratings like a video game for instant understanding
- **Plain-English Tooltips** — Hover any stat for a "Why it matters" explanation instead of math definitions
- **Trend Indicators** — Up/down arrows showing recent form vs season average
- **Luck Meter** — Visual comparison of actual vs expected stats (wOBA vs xwOBA)

### Models

- **PlayerLSTM**: 2-layer bidirectional LSTM with 8-head self-attention, LayerNorm, GELU
- **XGBoostPredictor**: Gradient-boosted trees with sequence flattening (mean/std/last/trend per feature)
- **EnsemblePredictor**: Weighted average or Ridge stacking meta-learner
- **Optuna HPT**: Bayesian hyperparameter optimization for both models

### Real Data Pipeline

- 15 features per game: batting avg, OBP, SLG, wOBA, barrel rate, exit velo, K%, BB%, sprint speed, hard hit rate, park factor, platoon advantage, days rest, launch angle, pull rate
- 4 prediction targets: hits, home runs, RBI, walks
- Temporal train/val/test split (no future data leakage)
- Parquet caching to avoid re-scraping Baseball Savant

## API Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| POST | `/v1/train` | Train LSTM + XGBoost |
| GET | `/v1/train/status` | Training progress |
| GET | `/v1/train/curves` | Training loss curves |
| POST | `/v1/predict/player` | Predict next-game stats |
| GET | `/v1/players/search?q=` | Search players by name |
| GET | `/v1/players/index` | Paginated player index |
| GET | `/v1/players/compare?ids=1,2` | Side-by-side player comparison |
| GET | `/v1/players/{id}` | Player profile + stats |
| GET | `/v1/players/{id}/predictions` | Player prediction history |
| GET | `/v1/games/live` | Live MLB games |
| GET | `/v1/games/today` | Today's schedule |
| GET | `/v1/games/{id}` | Game detail |
| GET | `/v1/games/{id}/predictions` | Player predictions for a game |
| GET | `/v1/schedule/range` | Schedule for date range |
| GET | `/v1/schedule/today` | Today's schedule |
| GET | `/v1/teams/` | All 30 MLB teams |
| GET | `/v1/predictions/daily` | Daily predictions hub |
| GET | `/v1/predictions/best-bets` | Top confidence predictions |
| POST | `/v1/attention/weights` | Attention heatmap |
| POST | `/v1/attention/feature-attribution` | Gradient feature importance |
| POST | `/v1/ensemble/predict` | Ensemble prediction |
| GET | `/v1/ensemble/weight-sensitivity` | Weight sweep analysis |
| GET | `/v1/model/evaluation` | Full evaluation + baselines |
| GET | `/v1/data/status` | Data freshness |
| POST | `/v1/data/refresh` | Trigger data refresh |
| POST | `/v1/tune` | Start Optuna tuning |
| GET | `/v1/tune/status` | Tuning progress |

## CI/CD

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | Push / PR to `master` | Backend import check (Python 3.13 + PostgreSQL), frontend build (Node 20) |
| `daily.yml` | Cron (10 AM UTC) + manual | Runs `refresh_daily_data()` to pull latest stats into PostgreSQL |

## Configuration

Backend settings are managed via `backend/core/config.py` using Pydantic. All environment variables use the `MLB_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `MLB_DATABASE_URL` | `sqlite+aiosqlite:///./mlb_analytics.db` | Async database URL |
| `MLB_DATABASE_URL_SYNC` | `sqlite:///./mlb_analytics.db` | Sync database URL |
| `MLB_CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `MLB_MODEL_DIR` | `models` | Model checkpoint directory |
| `MLB_DATA_SEASONS` | `[2023, 2024]` | Seasons to fetch data for |

Settings can also be loaded from a `.env` file.

## What I Learned

- **Feature engineering matters more than model architecture.** Rolling averages and trend features contributed more to prediction quality than switching between LSTM configurations.
- **Temporal splits are essential.** Random train/test splits in time-series data produce misleadingly optimistic metrics due to data leakage.
- **LSTM marginally beats XGBoost on sequential patterns**, but the gap is smaller than expected. XGBoost with engineered features is a strong baseline.
- **Baseball is inherently noisy.** Single-game predictions have high variance — this is a fundamental property of the sport, not a model limitation.
- **Stat context is everything for beginners.** Showing ".285 AVG" means nothing to a casual fan. Showing ".285 AVG (Great)" with a green badge and a percentile bar makes it instantly understandable.

## Limitations

- Predictions are for the next single game only. Game-to-game variance in baseball is high.
- Statcast features (barrel rate, exit velo) may be unavailable for some players/seasons.
- The platform uses free public data sources which may have scraping rate limits.
- Model accuracy should be compared against baselines (season average, recent average) — not evaluated in isolation.

## Project Structure

```
mlb-ai-analytics/
├── .github/workflows/          # CI (ci.yml) + daily data refresh (daily.yml)
├── frontend/                   # Next.js 14 dashboard
│   ├── app/                    # App Router pages
│   │   ├── dashboard/          # Dashboard, schedule, game detail, compare, players, predictions
│   │   ├── predict/            # Player prediction page
│   │   ├── models/             # Model comparison page
│   │   ├── attention/          # Attention visualizer
│   │   ├── ensemble/           # Ensemble lab
│   │   └── architecture/       # Architecture docs
│   ├── components/
│   │   ├── cards/              # MetricCard, GameCard, PlayerCard
│   │   ├── charts/             # TrainingCurves, AttentionHeatmap, WinProbability, etc.
│   │   ├── layout/             # ModernSidebar, Header
│   │   ├── predict/            # PlayerSearch, PredictionResult
│   │   ├── train/              # TrainControls, TrainProgress
│   │   ├── ui/                 # ContextBadge, PercentileBar, TrendIndicator, LuckMeter, StatTooltip
│   │   └── visuals/            # PlayerHeadshot
│   ├── hooks/                  # React Query hooks (useLiveGames, usePlayerSearch, usePrediction)
│   └── lib/                    # API client, types, stat helpers, constants
├── backend/                    # FastAPI application
│   ├── main.py                 # FastAPI entry point
│   ├── api/v1/                 # REST endpoints (train, predict, players, games, schedule, etc.)
│   ├── core/                   # Model service, evaluation, tuning, config
│   ├── db/                     # SQLAlchemy models (Player, PlayerStat, Game, Prediction), session
│   └── tasks/                  # Background data refresh
├── src/
│   ├── data/                   # Data pipeline, feature engineering
│   │   ├── pipeline.py         # Real MLB data fetching + DB seeding
│   │   ├── feature_builder.py  # Sliding window sequences from DB
│   │   └── cache_manager.py    # Parquet caching with TTL
│   ├── models/                 # PyTorch + XGBoost models
│   │   ├── predictor.py        # PlayerLSTM, GameTransformer
│   │   ├── xgboost_model.py    # XGBoost wrapper
│   │   ├── ensemble.py         # Ensemble methods
│   │   └── model_registry.py   # Training orchestration
│   └── services/
│       └── realtime.py         # Live game polling + win probability
├── docker-compose.yml          # Full stack deployment
└── training/                   # Standalone training scripts
```

## License

MIT License
