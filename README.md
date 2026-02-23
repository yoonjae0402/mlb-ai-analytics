# MLB AI Analytics Platform

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688)

An end-to-end ML platform that predicts MLB player performance using deep learning on real Statcast data. Features an 18-page Next.js dashboard with game win probability, pitcher matchup analysis, prediction accuracy tracking, and a beginner-friendly design system with contextual stat badges.

## What This Does

Predicts next-game batting stats (hits, HR, RBI, walks) for MLB players using a Bidirectional LSTM with attention trained on real game-by-game data from pybaseball and MLB Stats API. Computes game-level win probabilities via Pythagorean expectation, tracks prediction accuracy over time, and includes model comparison against XGBoost, LightGBM, and Ridge regression baselines, ensemble methods, and honest evaluation with confidence intervals.

## Architecture

```
pybaseball / MLB Stats API
        |
   PostgreSQL (players, player_stats, games, predictions)
        |
   Feature Engineering (26 features: Statcast + pitcher matchup + derived)
        |
   StandardScaler (fitted on training data only — no leakage)
        |
   Models: BiLSTM + Attention | XGBoost | LightGBM | Ridge/Lasso | Ensemble
        |
   FastAPI REST API (41 routes, async, background training)
        |
   Next.js 14 Dashboard (18 pages, React Query, Recharts, Tailwind)
```

## Tech Stack

| Layer        | Technologies                                                              |
| ------------ | ------------------------------------------------------------------------- |
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Recharts, React Query              |
| **Backend**  | FastAPI, SQLAlchemy (async + sync), Pydantic, Uvicorn                    |
| **Database** | PostgreSQL 16                                                             |
| **ML**       | PyTorch (BiLSTM + multi-head attention), XGBoost, LightGBM, Optuna      |
| **Data**     | pybaseball (Statcast, FanGraphs), MLB Stats API                          |
| **Deploy**   | Docker Compose                                                            |

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

### 18 Interactive Pages

1. **Home** — Feature overview and quick navigation cards
2. **Dashboard** — Live scores, system status, top predictions
3. **Schedule Calendar** — Week/month view with win probability bars on each game
4. **Game Predictions** — Click any game to see AI predictions for every player on both rosters, with win probability and team projections
5. **Prediction Hub** — All daily predictions sorted by stat, with confidence scores
6. **Leaderboard** — Top players ranked by composite predicted performance
7. **Player Predict** — Search any player, predict next-game stats with radar chart, context badges, and percentile bars
8. **Prediction Accuracy** — Track how predictions compare to actual results, calibration curves
9. **Player Index** — Browse all players with team/position filtering
10. **Player Detail** — Per-player profile with season totals, recent game log, and prediction history
11. **Pitcher Stats** — Search pitchers, view ERA, WHIP, K/9, BB/9, and season totals
12. **Compare Players** — Side-by-side stat comparison with context badges, trend indicators, and winner highlighting
13. **Model Comparison** — Train LSTM, XGBoost, LightGBM & Ridge side-by-side, live training curves
14. **Attention Visualizer** — Inspect what the LSTM model focuses on (heatmaps + gradient attribution)
15. **Ensemble Lab** — Weighted average vs stacking, weight sensitivity analysis
16. **Hyperparameter Tuning** — Optuna Bayesian optimization with trial history
17. **System Health** — Uptime, trained model metrics, database stats, last retrain
18. **Architecture & Docs** — System design, API reference, model documentation

### Beginner-Friendly Design System

- **Context Badges** — "Elite" (gold), "Great" (green), "Average" (gray), "Below Avg" (orange) pills next to every stat
- **Percentile Bars** — 0-99 ratings like a video game for instant understanding
- **Plain-English Tooltips** — Hover any stat for a "Why it matters" explanation instead of math definitions
- **Trend Indicators** — Up/down arrows showing recent form vs season average
- **Confidence Scores** — Color-coded confidence on predictions (green >= 70%, yellow >= 40%, gray < 40%)

### Models

- **PlayerLSTM**: 2-layer bidirectional LSTM with 8-head self-attention, LayerNorm, GELU — Input: (batch, 10, 26) → Output: (batch, 4)
- **XGBoostPredictor**: Gradient-boosted trees with sequence flattening (mean/std/last/trend per feature) — (n, 10, 26) → (n, 104)
- **LightGBMPredictor**: Leaf-wise gradient boosting with same sequence flattening as XGBoost
- **LinearPredictor**: Ridge or Lasso regression — simple interpretable baseline
- **EnsemblePredictor**: Weighted average or Ridge stacking meta-learner across all trained models
- **Monte Carlo Dropout**: 30 forward passes with dropout enabled for uncertainty estimation (90% CI)
- **Optuna HPT**: Bayesian hyperparameter optimization for LSTM and XGBoost
- **Champion/Challenger**: Daily retraining promotes new models only if validation MSE improves by >1%

### 26-Feature Pipeline

| Category                  | Features                                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Batter Core (15)**      | batting avg, OBP, SLG, wOBA, barrel rate, exit velo, launch angle, sprint speed, K%, BB%, hard hit %, pull %, park factor, platoon advantage, days rest |
| **Pitcher Matchup (5)**   | opposing ERA, opposing WHIP, opposing K/9, opposing BB/9, handedness advantage                                                                          |
| **Derived / Context (6)** | ISO, hot streak, BABIP, cold streak, home/away, opponent quality (win %)                                                                               |

- 4 prediction targets: hits, home runs, RBI, walks
- StandardScaler fitted on training data only (no leakage)
- Temporal train/val/test split (70% / 15% / 15%)
- Pitcher matchup matched by actual opponent from game schedule (not a date-scan heuristic)
- Parquet caching to avoid re-scraping Baseball Savant
- Uncertainty quantification: 90% confidence intervals on every LSTM prediction via MC Dropout

### Win Probability

- Pythagorean expectation (exponent 1.83) using aggregated player predictions
- Home field advantage adjustment (+0.3 projected runs)
- Missing batters filled with league-average slot contribution (~0.5 runs/slot) — not scaled extrapolation — to prevent skew when teams have different prediction coverage
- Displayed on schedule calendar, game detail, and individual game pages

## API Reference

| Method | Endpoint                            | Purpose                                           |
| ------ | ----------------------------------- | ------------------------------------------------- |
| GET    | `/health`                           | Health check                                      |
| POST   | `/v1/train`                         | Train LSTM + XGBoost (+ optional LightGBM/Linear) |
| GET    | `/v1/train/status`                  | Training progress                                 |
| GET    | `/v1/train/curves`                  | Training loss curves                              |
| POST   | `/v1/predict/player`                | Predict next-game stats                           |
| GET    | `/v1/players/search?q=`             | Search players by name                            |
| GET    | `/v1/players/index`                 | Paginated player index                            |
| GET    | `/v1/players/compare?ids=1,2`       | Side-by-side player comparison                    |
| GET    | `/v1/players/{id}`                  | Player profile + stats                            |
| GET    | `/v1/players/{id}/predictions`      | Player prediction history                         |
| GET    | `/v1/games/live`                    | Live MLB games                                    |
| GET    | `/v1/games/today`                   | Today's schedule                                  |
| GET    | `/v1/games/{id}`                    | Game detail                                       |
| GET    | `/v1/games/{id}/predictions`        | Player predictions for a game                     |
| GET    | `/v1/games/{id}/win-probability`    | Win probability + projections                     |
| GET    | `/v1/schedule/range`                | Schedule for date range                           |
| GET    | `/v1/schedule/today`                | Today's schedule                                  |
| GET    | `/v1/teams/`                        | All 30 MLB teams                                  |
| GET    | `/v1/predictions/daily`             | Daily predictions hub                             |
| GET    | `/v1/predictions/best-bets`         | Top confidence predictions                        |
| GET    | `/v1/leaderboard`                   | Player leaderboard                                |
| GET    | `/v1/pitchers/search`               | Search pitchers                                   |
| GET    | `/v1/pitchers/{id}/stats`           | Pitcher stats (ERA, WHIP, K/9, BB/9)             |
| GET    | `/v1/accuracy/summary`              | Prediction accuracy summary                       |
| GET    | `/v1/accuracy/by-player/{id}`       | Per-player accuracy                               |
| GET    | `/v1/accuracy/calibration`          | Calibration curve data                            |
| POST   | `/v1/accuracy/backfill`             | Backfill prediction results                       |
| POST   | `/v1/attention/weights`             | Attention heatmap                                 |
| POST   | `/v1/attention/feature-attribution` | Gradient feature importance                       |
| POST   | `/v1/ensemble/predict`              | Ensemble prediction                               |
| GET    | `/v1/ensemble/weight-sensitivity`   | Weight sweep analysis                             |
| GET    | `/v1/model/evaluation`              | Full evaluation + baselines                       |
| GET    | `/v1/model/metrics`                 | Current model metrics                             |
| GET    | `/v1/data/status`                   | Data freshness                                    |
| POST   | `/v1/data/refresh`                  | Trigger data refresh                              |
| POST   | `/v1/tune`                          | Start Optuna tuning                               |
| GET    | `/v1/tune/status`                   | Tuning progress                                   |
| GET    | `/v1/system/health`                 | System health + model versions                    |
| GET    | `/v1/scheduler/status`              | Scheduler status                                  |
| POST   | `/v1/scheduler/run`                 | Trigger manual scheduler run                      |
| POST   | `/v1/baseline/generate`             | Generate baseline predictions for all players     |

## CI/CD

| Workflow    | Trigger                   | What it does                                                              |
| ----------- | ------------------------- | ------------------------------------------------------------------------- |
| `ci.yml`    | Push / PR to `master`     | Backend import check (Python 3.13 + PostgreSQL), frontend build (Node 20) |
| `daily.yml` | Cron (10 AM UTC) + manual | Runs `refresh_daily_data()` to pull latest stats into PostgreSQL          |

## Configuration

Backend settings are managed via `backend/core/config.py` using Pydantic. All environment variables use the `MLB_` prefix:

| Variable                | Default                                                    | Description                |
| ----------------------- | ---------------------------------------------------------- | -------------------------- |
| `MLB_DATABASE_URL`      | `postgresql+asyncpg://mlb:mlb@localhost/mlb_analytics`     | Async database URL         |
| `MLB_DATABASE_URL_SYNC` | `postgresql+psycopg2://mlb:mlb@localhost/mlb_analytics`    | Sync database URL          |
| `MLB_CORS_ORIGINS`      | `["http://localhost:3000"]`                                | Allowed CORS origins       |
| `MLB_MODEL_DIR`         | `models`                                                   | Model checkpoint directory |
| `MLB_DATA_SEASONS`      | `[2023, 2024]`                                             | Seasons to fetch data for  |
| `MLB_AUTO_SEED`         | `false`                                                    | Auto-seed DB on startup    |

Settings can also be loaded from a `.env` file.

---

## Prediction Methodology

This section documents the complete prediction logic so developers can understand, audit, and reproduce the models.

### Overview

The platform predicts **per-game batting stats** (hits, HR, RBI, walks) for individual MLB players using a sequence model trained on rolling game-by-game data. Game-level win probability is derived from these player predictions using Pythagorean expectation.

**Prediction targets:** 4 outputs — `hits`, `home_runs`, `rbi`, `walks`

---

### 1. Data Sources

| Source | Data Fetched | Library |
|--------|-------------|---------|
| **Baseball Savant (Statcast)** | Exit velocity, barrel rate, launch angle, sprint speed, hard hit %, pull % | `pybaseball.statcast_batter()` |
| **FanGraphs** | wOBA, K%, BB%, BABIP, ISO, platoon splits | `pybaseball.batting_stats()` |
| **MLB Stats API** | Game schedule, lineup, game results, probable pitchers | `statsapi.schedule()`, `statsapi.boxscore()` |
| **PostgreSQL DB** | Pre-processed and cached player stats, games, predictions | SQLAlchemy ORM |

---

### 2. Feature Engineering (26 Features)

Features are computed per game appearance and assembled into **sliding window sequences of 10 consecutive games** (shape `[N, 10, 26]`).

#### Batter Core (15 features)
| Feature | Source | Description |
|---------|---------|-------------|
| `batting_avg` | FanGraphs | Rolling 10-game batting average |
| `on_base_pct` (OBP) | FanGraphs | (H+BB+HBP)/(AB+BB+HBP+SF) |
| `slugging_pct` (SLG) | FanGraphs | Total bases / AB |
| `woba` | FanGraphs | Weighted On-Base Average (linear weights model) |
| `barrel_rate` | Statcast | % of batted balls with ideal exit velo + angle |
| `exit_velocity` | Statcast | Average exit velocity (mph) on contact |
| `launch_angle` | Statcast | Average launch angle (degrees) |
| `sprint_speed` | Statcast | Feet per second during max effort runs |
| `k_rate` | FanGraphs | Strikeout rate (K/PA) |
| `bb_rate` | FanGraphs | Walk rate (BB/PA) |
| `hard_hit_rate` | Statcast | % batted balls at 95+ mph exit velocity |
| `pull_rate` | Statcast | % batted balls pulled to pull-side |
| `park_factor` | FanGraphs | Run environment adjustment for ballpark |
| `platoon_advantage` | MLB Stats API | +1 if batter/pitcher handedness mismatch (favors batter) |
| `days_rest` | Schedule | Days since last game appearance |

#### Pitcher Matchup (5 features)
| Feature | Source | Description |
|---------|---------|-------------|
| `opp_era` | DB / MLB API | Starting pitcher's season ERA |
| `opp_whip` | DB / MLB API | Starting pitcher's WHIP (H+BB per inning) |
| `opp_k_per_9` | DB / MLB API | Starting pitcher's K/9 |
| `opp_bb_per_9` | DB / MLB API | Starting pitcher's BB/9 |
| `opp_handedness_adv` | MLB API | +1 if same hand (pitcher advantage), -1 if opposite |

**Important:** Pitcher matching uses the **actual opponent** from the game schedule, looked up O(1) from a pre-built `{(team, date): opponent}` dictionary. This avoids the common bug of scanning all teams on a date.

#### Derived / Context (6 features)
| Feature | Formula | Description |
|---------|---------|-------------|
| `iso` | SLG - AVG | Isolated Power (extra bases per AB) |
| `hot_streak` | Last 7 games ≥ .350 AVG | Binary hot streak indicator |
| `cold_streak` | Last 7 games ≤ .150 AVG | Binary cold streak indicator |
| `babip` | (H - HR) / (AB - K - HR + SF) | Batting Average on Balls in Play |
| `is_home` | Schedule | 1 = home team, 0 = away |
| `opp_quality` | Team win% | Opponent quality based on current win percentage |

#### Preprocessing
- All 26 features are **StandardScaler** normalized — scaler is fit **on training data only** to prevent data leakage
- Missing Statcast features default to per-position league averages (not dropped)

---

### 3. Model Architecture

#### Primary: BiLSTM + Multi-Head Attention (`PlayerLSTM`)
```
Input: (batch, 10, 26)
  → Linear projection: 26 → hidden_size
  → 2-layer Bidirectional LSTM: hidden_size × 2
  → 8-head Self-Attention + LayerNorm
  → Last timestep → GELU → Dropout(0.3) → Linear(4)
Output: (batch, 4) [hits, hr, rbi, walks]
```
- Hidden size: 128 (default, tunable via Optuna)
- Learning rate: 1e-3 (Adam, cosine annealing)
- Epochs: 50 with early stopping (patience=10)

#### Ensemble (`EnsemblePredictor`)
Combines predictions from LSTM + XGBoost + LightGBM + Ridge:
- **Weighted average**: `Σ(weight_i × prediction_i)` — default equal weights
- **Stacking**: Ridge regression meta-learner trained on held-out validation predictions

Weights are optimized by minimizing validation MSE on a held-out split.

---

### 4. Starting Lineup Prediction

The platform does **not** attempt to predict batting order from scratch. Instead:

1. **Probable pitchers** come from MLB Stats API `schedule()` endpoint (official probable starters)
2. **Batting lineups** on game day come from `statsapi.boxscore()` for games in progress / completed
3. For **pre-game lineup estimation**, the system uses: the most recent lineup from the team's last game (from DB), filtered by active roster (not on IL)
4. Players are flagged as "Probable" (in recent lineup), "Questionable" (limited recent ABs), or "Day-to-Day" (IL status from MLB API)

---

### 5. Projected Runs Formula

```
projected_runs(team) = Σ(player_woba_pred × PA_estimate) × run_weight × park_factor × era_adjustment
```

Step-by-step:
1. **Per-player projected wOBA** from the LSTM ensemble
2. **PA estimate**: average PAs per slot by batting order position (leadoff ~4.5, 9th ~3.5)
3. **Run weight**: wOBA ÷ 1.25 (approximates runs per PA from wOBA linear weights)
4. **Park factor**: venue-specific run environment (from FanGraphs park factor table, stored per game)
5. **ERA adjustment**: `max(0.7, min(1.3, league_avg_era / starter_era))` — better pitchers reduce run projection

**Missing players**: Slots with no prediction data use `league_avg_per_slot = 4.5 runs / 9 batters = 0.5 runs/slot`. This prevents skew when one team has fewer predictions than the other.

**Home field advantage**: +0.3 projected runs added to the home team total.

---

### 6. Win Probability Calculation

```python
def pythagorean_win_prob(runs_scored, runs_allowed, exponent=1.83):
    return runs_scored**exponent / (runs_scored**exponent + runs_allowed**exponent)
```

- **Pre-game**: Uses projected runs from Section 5 above
- **In-game**: Win probability updates after each inning using actual runs scored + projected remaining innings
- **Exponent 1.83**: This is the empirically validated exponent (Davenport/Smyth refinement of Bill James's original 2.0)

**Inning-by-inning WP history** is stored in the `games.wp_history` column as a JSON array and displayed as a line chart.

---

### 7. Uncertainty & Confidence Intervals

**Monte Carlo Dropout (MC Dropout):**
- LSTM model runs **30 forward passes** with dropout enabled at inference time
- Standard deviation of the 30 predictions gives per-stat uncertainty
- **90% confidence interval**: `prediction ± 1.645 × std_dev`

**Confidence Score** = `1 / (1 + avg_std_dev)` — closer to 1 means more certain

---

### 8. Evaluation Methodology (Honest)

To avoid misleadingly optimistic metrics:

1. **Temporal split**: Training 70%, validation 15%, test 15% — always chronological (no random split which would leak future data)
2. **Baselines compared**: Each model is compared against (a) season average, (b) rolling 7-game average, (c) last-game stats
3. **Bootstrap confidence intervals**: 1000 bootstrap samples on the test set to get 95% CI on MAE and R²
4. **Prediction tracking**: Every prediction is stored in `predictions` table; `accuracy/backfill` compares against actual box scores pulled from MLB Stats API

---

### 9. Limitations

- Single-game predictions in baseball have **high inherent variance** (BABIP noise, small sample effects)
- Statcast features (barrel rate, exit velocity) require sufficient plate appearances to be meaningful
- Lineup predictions before official lineups are released are estimates based on historical tendencies
- Public API rate limits may cause scraping delays; Parquet caching mitigates this
- The model was trained on 2023–2024 data; performance on players with limited recent history may be lower

---

## What I Learned

- **Feature engineering matters more than model architecture.** Rolling averages and trend features contributed more to prediction quality than switching between LSTM configurations.
- **Temporal splits are essential.** Random train/test splits in time-series data produce misleadingly optimistic metrics due to data leakage.
- **Pitcher matchup features add signal.** Opposing pitcher ERA, WHIP, and K/9 improve predictions beyond batter-only features. Matching the *actual* opponent (not any team on the same date) is critical for correctness.
- **LSTM marginally beats XGBoost on sequential patterns**, but the gap is smaller than expected. XGBoost with engineered features is a strong baseline.
- **Win probability requires balanced coverage.** When one team has fewer player predictions than the other, run estimates must be normalized to league average per missing slot — not scaled up — to prevent skewed probabilities.
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
├── frontend/                   # Next.js 14 dashboard (18 pages)
│   ├── app/                    # App Router pages
│   │   ├── dashboard/          # Dashboard, schedule, game detail, predictions, leaderboard,
│   │   │                       # accuracy, players, player detail, pitchers, compare, system
│   │   ├── predict/            # Player prediction page
│   │   ├── models/             # Model comparison page
│   │   ├── attention/          # Attention visualizer
│   │   ├── ensemble/           # Ensemble lab
│   │   ├── tuning/             # Hyperparameter tuning
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
├── backend/                    # FastAPI application (41 routes)
│   ├── main.py                 # FastAPI entry point
│   ├── api/v1/                 # REST endpoints (train, predict, players, games, schedule,
│   │                           # pitchers, accuracy, leaderboard, ensemble, tuning, system, etc.)
│   ├── core/                   # Model service, evaluation, tuning, config
│   ├── db/                     # SQLAlchemy models (Player, PlayerStat, Game, Prediction), session
│   ├── services/               # Game predictor (win probability), baseline predictor
│   └── tasks/                  # Background data refresh, prediction backfill, daily retrain
├── src/
│   ├── data/                   # Data pipeline, feature engineering
│   │   ├── pipeline.py         # Real MLB data fetching + DB seeding (batters + pitchers)
│   │   ├── feature_builder.py  # 26-feature sliding window sequences + StandardScaler
│   │   └── cache_manager.py    # Parquet caching with TTL
│   └── models/                 # PyTorch + tree models
│       ├── predictor.py        # PlayerLSTM (BiLSTM + attention), MC Dropout
│       ├── xgboost_model.py    # XGBoost wrapper with sequence flattening
│       ├── lightgbm_model.py   # LightGBM wrapper
│       ├── linear_model.py     # Ridge/Lasso linear baseline
│       ├── ensemble.py         # Ensemble methods (weighted avg, stacking)
│       └── model_registry.py   # Training orchestration
├── docker-compose.yml          # Full stack deployment (PostgreSQL + backend + frontend)
└── training/                   # Standalone training scripts
```

## License

MIT License
