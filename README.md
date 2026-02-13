# MLB AI Analytics Platform

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688)

An end-to-end ML platform that predicts MLB player performance using deep learning on real Statcast data. Features a 15-page Next.js dashboard with game win probability, pitcher matchup analysis, prediction accuracy tracking, and a beginner-friendly design system with contextual stat badges.

## What This Does

Predicts next-game batting stats (hits, HR, RBI, walks) for MLB players using a Bidirectional LSTM with attention trained on real game-by-game data from pybaseball and MLB Stats API. Computes game-level win probabilities via Pythagorean expectation, tracks prediction accuracy over time, and includes model comparison against XGBoost baselines, ensemble methods, and honest evaluation with confidence intervals.

## Architecture

```
pybaseball / MLB Stats API
        |
   PostgreSQL (players, player_stats, games, predictions)
        |
   Feature Engineering (22 features: Statcast + pitcher matchup + derived)
        |
   StandardScaler (fitted on training data only)
        |
   Models: BiLSTM + Attention | XGBoost | Ensemble
        |
   FastAPI REST API (39 routes, async, background training)
        |
   Next.js 14 Dashboard (15 pages, React Query, Recharts, Tailwind)
```

## Tech Stack

| Layer        | Technologies                                                |
| ------------ | ----------------------------------------------------------- |
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Recharts, React Query |
| **Backend**  | FastAPI, SQLAlchemy (async + sync), Pydantic, Uvicorn       |
| **Database** | PostgreSQL 16                                               |
| **ML**       | PyTorch (BiLSTM + multi-head attention), XGBoost, Optuna    |
| **Data**     | pybaseball (Statcast, FanGraphs), MLB Stats API             |
| **Deploy**   | Docker Compose                                              |

## Getting Started

### Docker (recommended)

```bash
docker-compose up
```

This starts PostgreSQL, the FastAPI backend (port 8000), and the Next.js frontend (port 3000). The backend automatically seeds the database with MLB data on first run.

### Manual Setup

```bash
# Backend
pip install -r backend/requirements.txt

# Start PostgreSQL (ensure it's running on port 5432)
# Create database: createdb mlb_analytics

# Seed database with real MLB data
python3 -c "from src.data.pipeline import MLBDataPipeline; MLBDataPipeline().seed_database([2025])"

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

### 15 Interactive Pages

1. **Home** — Feature overview and quick navigation cards
2. **Dashboard** — Live scores, system status, top predictions
3. **Schedule Calendar** — Week/month view with win probability bars on each game
4. **Game Predictions** — Click any game to see AI predictions for every player on both rosters, with win probability and team projections
5. **Prediction Hub** — All daily predictions sorted by stat, with confidence scores
6. **Leaderboard** — Top players ranked by composite predicted performance
7. **Player Predict** — Search any player, predict next-game stats with radar chart, context badges, and percentile bars
8. **Prediction Accuracy** — Track how predictions compare to actual results, calibration curves
9. **Player Index** — Browse all players with team/position filtering
10. **Pitcher Stats** — Search pitchers, view ERA, WHIP, K/9, BB/9, and season totals
11. **Compare Players** — Side-by-side stat comparison with context badges, trend indicators, and winner highlighting
12. **Model Comparison** — Train LSTM & XGBoost side-by-side, live training curves
13. **Attention Visualizer** — Inspect what the LSTM model focuses on (heatmaps + gradient attribution)
14. **Ensemble Lab** — Weighted average vs stacking, weight sensitivity analysis
15. **Architecture & Docs** — System design, API reference, model documentation

### Beginner-Friendly Design System

- **Context Badges** — "Elite" (gold), "Great" (green), "Average" (gray), "Below Avg" (orange) pills next to every stat
- **Percentile Bars** — 0-99 ratings like a video game for instant understanding
- **Plain-English Tooltips** — Hover any stat for a "Why it matters" explanation instead of math definitions
- **Trend Indicators** — Up/down arrows showing recent form vs season average
- **Confidence Scores** — Color-coded confidence on predictions (green >= 70%, yellow >= 40%, gray < 40%)

### Models

- **PlayerLSTM**: 2-layer bidirectional LSTM with 8-head self-attention, LayerNorm, GELU — Input: (batch, 10, 26) -> Output: (batch, 4)
- **XGBoostPredictor**: Gradient-boosted trees with sequence flattening (mean/std/last/trend per feature) — (n, 10, 26) -> (n, 104)
- **EnsemblePredictor**: Weighted average or Ridge stacking meta-learner
- **Monte Carlo Dropout**: 30 forward passes with dropout enabled for uncertainty estimation (90% CI)
- **Optuna HPT**: Bayesian hyperparameter optimization for both models

### 26-Feature Pipeline

| Category                    | Features                                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Batter Core (15)**        | batting avg, OBP, SLG, wOBA, barrel rate, exit velo, launch angle, sprint speed, K%, BB%, hard hit %, pull %, park factor, platoon advantage, days rest |
| **Pitcher Matchup (5)**     | opposing ERA, opposing WHIP, opposing K/9, opposing BB/9, handedness advantage                                                                          |
| **Derived / Context (6)**   | ISO, hot streak, BABIP, cold streak, home/away, opponent quality (win %)                                                                                |

- 4 prediction targets: hits, home runs, RBI, walks
- StandardScaler fitted on training data only (no leakage)
- Temporal train/val/test split
- Parquet caching to avoid re-scraping Baseball Savant
- Uncertainty quantification: confidence intervals on every prediction

### Win Probability

- Pythagorean expectation (exponent 1.83) using aggregated player predictions
- Home field advantage adjustment (+0.3 projected runs)
- Displayed on schedule calendar, game detail, and individual game pages

## API Reference

| Method | Endpoint                            | Purpose                        |
| ------ | ----------------------------------- | ------------------------------ |
| GET    | `/health`                           | Health check                   |
| POST   | `/v1/train`                         | Train LSTM + XGBoost           |
| GET    | `/v1/train/status`                  | Training progress              |
| GET    | `/v1/train/curves`                  | Training loss curves           |
| POST   | `/v1/predict/player`                | Predict next-game stats        |
| GET    | `/v1/players/search?q=`             | Search players by name         |
| GET    | `/v1/players/index`                 | Paginated player index         |
| GET    | `/v1/players/compare?ids=1,2`       | Side-by-side player comparison |
| GET    | `/v1/players/{id}`                  | Player profile + stats         |
| GET    | `/v1/players/{id}/predictions`      | Player prediction history      |
| GET    | `/v1/games/live`                    | Live MLB games                 |
| GET    | `/v1/games/today`                   | Today's schedule               |
| GET    | `/v1/games/{id}`                    | Game detail                    |
| GET    | `/v1/games/{id}/predictions`        | Player predictions for a game  |
| GET    | `/v1/games/{id}/win-probability`    | Win probability + projections  |
| GET    | `/v1/schedule/range`                | Schedule for date range        |
| GET    | `/v1/schedule/today`                | Today's schedule               |
| GET    | `/v1/teams/`                        | All 30 MLB teams               |
| GET    | `/v1/predictions/daily`             | Daily predictions hub          |
| GET    | `/v1/predictions/best-bets`         | Top confidence predictions     |
| GET    | `/v1/leaderboard`                   | Player leaderboard             |
| GET    | `/v1/pitchers/search`               | Search pitchers                |
| GET    | `/v1/pitchers/{id}/stats`           | Pitcher stats                  |
| GET    | `/v1/accuracy/summary`              | Prediction accuracy summary    |
| GET    | `/v1/accuracy/by-player/{id}`       | Per-player accuracy            |
| GET    | `/v1/accuracy/calibration`          | Calibration curve data         |
| POST   | `/v1/accuracy/backfill`             | Backfill prediction results    |
| POST   | `/v1/attention/weights`             | Attention heatmap              |
| POST   | `/v1/attention/feature-attribution` | Gradient feature importance    |
| POST   | `/v1/ensemble/predict`              | Ensemble prediction            |
| GET    | `/v1/ensemble/weight-sensitivity`   | Weight sweep analysis          |
| GET    | `/v1/model/evaluation`              | Full evaluation + baselines    |
| GET    | `/v1/data/status`                   | Data freshness                 |
| POST   | `/v1/data/refresh`                  | Trigger data refresh           |
| POST   | `/v1/tune`                          | Start Optuna tuning            |
| GET    | `/v1/tune/status`                   | Tuning progress                |

## CI/CD

| Workflow    | Trigger                   | What it does                                                              |
| ----------- | ------------------------- | ------------------------------------------------------------------------- |
| `ci.yml`    | Push / PR to `master`     | Backend import check (Python 3.13 + PostgreSQL), frontend build (Node 20) |
| `daily.yml` | Cron (10 AM UTC) + manual | Runs `refresh_daily_data()` to pull latest stats into PostgreSQL          |

## Configuration

Backend settings are managed via `backend/core/config.py` using Pydantic. All environment variables use the `MLB_` prefix:

| Variable                | Default                                  | Description                |
| ----------------------- | ---------------------------------------- | -------------------------- |
| `MLB_DATABASE_URL`      | `sqlite+aiosqlite:///./mlb_analytics.db` | Async database URL         |
| `MLB_DATABASE_URL_SYNC` | `sqlite:///./mlb_analytics.db`           | Sync database URL          |
| `MLB_CORS_ORIGINS`      | `["http://localhost:3000"]`              | Allowed CORS origins       |
| `MLB_MODEL_DIR`         | `models`                                 | Model checkpoint directory |
| `MLB_DATA_SEASONS`      | `[2023, 2024]`                           | Seasons to fetch data for  |
| `MLB_AUTO_SEED`         | `false`                                  | Auto-seed DB on startup    |

Settings can also be loaded from a `.env` file.

## What I Learned

- **Feature engineering matters more than model architecture.** Rolling averages and trend features contributed more to prediction quality than switching between LSTM configurations.
- **Temporal splits are essential.** Random train/test splits in time-series data produce misleadingly optimistic metrics due to data leakage.
- **Pitcher matchup features add signal.** Opposing pitcher ERA, WHIP, and K/9 improve predictions beyond batter-only features.
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
├── frontend/                   # Next.js 14 dashboard (15 pages)
│   ├── app/                    # App Router pages
│   │   ├── dashboard/          # Dashboard, schedule, game detail, predictions, leaderboard,
│   │   │                       # accuracy, players, pitchers, compare
│   │   ├── predict/            # Player prediction page
│   │   ├── models/             # Model comparison page
│   │   ├── attention/          # Attention visualizer
│   │   ├── ensemble/           # Ensemble lab
│   │   └── architecture/       # Architecture docs
│   ├── components/
│   │   ├── cards/              # MetricCard, GameCard, PlayerCard
│   │   ├── charts/             # TrainingCurves, AttentionHeatmap, RadarChart, etc.
│   │   ├── layout/             # ModernSidebar, Header
│   │   ├── predict/            # PlayerSearch, PredictionResult
│   │   ├── train/              # TrainControls, TrainProgress
│   │   ├── ui/                 # ContextBadge, PercentileBar, TrendIndicator, StatTooltip
│   │   └── visuals/            # PlayerHeadshot
│   ├── hooks/                  # React Query hooks (useLiveGames, usePlayerSearch, usePrediction)
│   └── lib/                    # API client, types, stat helpers, constants
├── backend/                    # FastAPI application (39 routes)
│   ├── main.py                 # FastAPI entry point
│   ├── api/v1/                 # REST endpoints (train, predict, players, games, schedule,
│   │                           # pitchers, accuracy, leaderboard, etc.)
│   ├── core/                   # Model service, evaluation, tuning, config
│   ├── db/                     # SQLAlchemy models (Player, PlayerStat, Game, Prediction), session
│   ├── services/               # Game predictor (win probability)
│   └── tasks/                  # Background data refresh, prediction backfill
├── src/
│   ├── data/                   # Data pipeline, feature engineering
│   │   ├── pipeline.py         # Real MLB data fetching + DB seeding (batters + pitchers)
│   │   ├── feature_builder.py  # 22-feature sliding window sequences + StandardScaler
│   │   └── cache_manager.py    # Parquet caching with TTL
│   └── models/                 # PyTorch + XGBoost models
│       ├── predictor.py        # PlayerLSTM (BiLSTM + attention)
│       ├── xgboost_model.py    # XGBoost wrapper
│       ├── ensemble.py         # Ensemble methods
│       └── model_registry.py   # Training orchestration
├── docker-compose.yml          # Full stack deployment (PostgreSQL + backend + frontend)
└── training/                   # Standalone training scripts
```

## License

MIT License
