# MLB AI Analytics — Implementation Plan

## Current Focus: Docker Database Auto-Seeding + Baseline Predictions

### Problem
When running with Docker Compose, the backend connects to a **fresh PostgreSQL container with no data**. Tables are created on startup via `init_db()`, but no players, stats, or predictions exist. The `seed_database` script was only run locally against SQLite.

### Root Cause
- `docker-compose.yml` starts a fresh PostgreSQL with empty tables
- `backend/main.py` creates tables but has no data seeding step
- Baseline prediction generation finds 0 players → generates 0 predictions

---

## Proposed Changes

### 1. Docker Entrypoint Script
**[NEW] `backend/entrypoint.sh`**
- Auto-seeds 2025 MLB + AAA data on first startup
- Generates baseline predictions (weighted historical averages, no ML model needed)
- Only runs when `MLB_AUTO_SEED=true` (Docker only)
- Starts `uvicorn` after seeding completes

### 2. Backend Dockerfile
**[MODIFY] `backend/Dockerfile`**
- Copy `entrypoint.sh` into image
- Set as `ENTRYPOINT`

### 3. Docker Compose
**[MODIFY] `docker-compose.yml`**
- Add `MLB_AUTO_SEED: "true"` to backend environment

### 4. Dependencies
**[MODIFY] `backend/requirements.txt`**
- Add `aiosqlite>=0.20.0` for local SQLite dev support

---

## Baseline Prediction System (Already Implemented)
- **`backend/services/baseline_predictor.py`**: Generates predictions from weighted historical averages
- **No ML model training required** — uses last 30 games with recency weighting
- **No user input required** — auto-generates on startup + on-the-fly per request
- **`backend/api/v1/games.py`**: Falls back to baseline predictions when no trained model exists

---

## Data Pipeline Config
- **Seasons**: `[2025, 2026]` (2025 completed, 2026 current)
- **Levels**: MLB (`sportId=1`) + Triple-A (`sportId=11`) only
- **Players**: ~1,600 (1,470 MLB + 130 AAA)
- **Stats**: ~76,000 game-level entries
- **Predictions**: ~727 baseline predictions (players with ≥3 games)

---

## Previously Completed

### Frontend - Design & Features
- [x] Game Prediction Page (`frontend/app/dashboard/game/[gameId]/page.tsx`)
- [x] Interactive Schedule linking to game predictions
- [x] Player Comparison Tool (`frontend/app/dashboard/compare/page.tsx`)
- [x] Context Badges, TrendIndicator, InfoTooltip components
- [x] Search robustness for empty states

### Backend - APIs
- [x] `GET /v1/games/{game_id}` — game details
- [x] `GET /v1/games/{game_id}/predictions` — player predictions with auto-generation
- [x] `POST /v1/baseline/generate` — bulk baseline prediction generation
- [x] Team matching fix (abbreviation + ID-based lookup)

### Database / Data
- [x] 2025 season data seeded (MLB + AAA)
- [x] Baseline predictions generated for all eligible players

---

## Verification Plan

### Docker
- Run `docker compose up --build`
- Check backend logs for seeding progress
- Open `http://localhost:3000` → verify players appear
- Click a game in schedule → verify predictions show up

### Local Dev
- `uvicorn backend.main:app --reload` with SQLite
- Verify auto-prediction on startup
- Test game prediction page

---

## Phase 2: Analytics Platform Improvements

### Current State (Completed)
- 12 frontend pages, 16 API route modules, LSTM + XGBoost models
- 2,538 players, 106K stats, 1,115 baseline predictions (Docker)
- Docker Compose full-stack deployment working
- Baseline prediction fallback system

### Priority 1 — Pitcher Matchup Integration
**Impact: Highest — pitching drives ~50% of outcomes**

**[MODIFY] `src/data/pipeline.py`**
- Ingest pitcher stats (ERA, WHIP, K-rate, BB-rate, FIP) into `player_stats`
- Fetch pitcher handedness and platoon splits
- Store opposing pitcher ID per game in `player_stats` or new join table

**[MODIFY] `src/data/feature_builder.py`**
- Add opponent pitcher features to sequence: ERA, WHIP, K/9, handedness match
- Compute batter-vs-pitcher-handedness splits (L/R advantage)
- Increase feature count from 15 → ~22

**[MODIFY] `src/models/predictor.py`**
- Update `PlayerLSTM` input_size to handle new features
- Consider dual-encoder: one for batter sequence, one for pitcher context

**[NEW] `backend/api/v1/pitchers.py`**
- `GET /v1/pitchers/search` — search pitchers
- `GET /v1/pitchers/{id}/stats` — pitcher profile with recent performance

### Priority 2 — Feature Scaling (Critical Bug Fix)
**Impact: High — LSTM convergence severely hurt without normalization**

**[MODIFY] `src/data/feature_builder.py`**
- Add `StandardScaler` fit on training data
- Transform all features before sequence creation
- Save scaler parameters for inference-time use
- Ensure inverse transform for interpretability

**[MODIFY] `src/models/model_registry.py`**
- Persist scaler alongside model checkpoint
- Load scaler during prediction

### Priority 3 — Game-Level Predictions & Win Probability
**Impact: High — this is what users actually care about**

**[NEW] `backend/services/game_predictor.py`**
- Aggregate player-level predictions → projected team runs
- Simple win probability model (Pythagorean expectation or logistic regression)
- Compare projected totals to Vegas lines (over/under)

**[MODIFY] `backend/api/v1/games.py`**
- Add `GET /v1/games/{game_id}/win-probability` endpoint
- Return: home_win_pct, away_win_pct, projected_runs_home, projected_runs_away
- Include confidence interval

**[MODIFY] `frontend/app/dashboard/game/[gameId]/page.tsx`**
- Display win probability bar chart
- Show projected run totals for each team
- Add "Game Outlook" summary card

### Priority 4 — Prediction Accuracy Tracking
**Impact: High — proves the model works (or doesn't)**

**[NEW] `backend/tasks/backfill_results.py`**
- After games complete, fetch actual stats from MLB API
- Compare predictions vs actuals, populate `prediction_results` table
- Compute per-prediction MSE and store it

**[NEW] `backend/api/v1/accuracy.py`**
- `GET /v1/accuracy/summary` — overall model accuracy (hit rate, avg error)
- `GET /v1/accuracy/by-player/{id}` — per-player prediction track record
- `GET /v1/accuracy/calibration` — calibration curve data

**[NEW] `frontend/app/dashboard/accuracy/page.tsx`**
- Accuracy dashboard: overall hit rate, error distribution
- Calibration curve chart
- "Best predicted" and "worst predicted" player lists
- Timeline of model accuracy over the season

### Priority 5 — Advanced Feature Engineering
**Impact: Medium-High — better features = better predictions**

**[MODIFY] `src/data/feature_builder.py`**
Add these derived features:

| Feature | Formula | Why |
|---------|---------|-----|
| ISO | SLG - BA | Power isolation |
| BABIP | (H - HR) / (AB - K - HR + SF) | Luck vs skill |
| wRC+ | Fetch from FanGraphs via pybaseball | Park/league adjusted |
| Hot streak | 1 if BA > .350 last 7 games | Momentum |
| Cold streak | 1 if BA < .150 last 7 games | Slump detection |
| Home/away | Binary indicator | Split performance |
| Days since rest | Count games since last off-day | Fatigue proxy |
| Opponent quality | Opponent team winning % | Schedule strength |

### Priority 6 — Uncertainty Quantification
**Impact: Medium — makes predictions trustworthy**

**[MODIFY] `src/models/predictor.py`**
- Add Monte Carlo Dropout: run N forward passes with dropout enabled
- Return mean prediction + std as confidence interval
- Confidence = 1 / (1 + std)

**[MODIFY] `backend/api/v1/schemas.py`**
- Add `confidence_interval_low` and `confidence_interval_high` to PredictionResponse

**[MODIFY] `frontend/app/predict/page.tsx`**
- Show confidence interval bars on prediction cards
- Color-code by uncertainty (green = tight, yellow = moderate, red = wide)

### Priority 7 — Model Diversity & Ensemble
**Impact: Medium — more diverse models = better ensemble**

**[NEW] `src/models/lightgbm_model.py`**
- LightGBM regressor with same feature flattening as XGBoost
- Faster training, different regularization approach

**[NEW] `src/models/linear_model.py`**
- Ridge/Lasso regression baseline
- Important for ensemble diversity and as honest baseline

**[MODIFY] `src/models/ensemble.py`**
- Support 3+ base models in weighted average and stacking
- Add automatic weight optimization via cross-validation
- Report ensemble diversity metrics

### Priority 8 — Missing Frontend Pages
**Impact: Medium — completes the product**

**[NEW] `frontend/app/dashboard/player/[playerId]/page.tsx`**
- Career stats overview with season-by-season table
- Prediction history chart (predicted vs actual over time)
- Stat trends with sparklines
- Similar players section

**[NEW] `frontend/app/dashboard/accuracy/page.tsx`**
- Model accuracy dashboard (see Priority 4)

**[NEW] `frontend/app/tuning/page.tsx`**
- Start/monitor Optuna hyperparameter tuning
- Display best params, trial history, optimization curve
- API already exists: `POST /v1/tune`, `GET /v1/tune/status`

### Priority 9 — Evaluation Rigor
**Impact: Medium — proves analytical credibility**

**[MODIFY] `backend/core/evaluation.py`**
- Add 5-fold time-series cross-validation (expanding window)
- Add metrics: MAE, RMSE, MAPE, R² per target
- Add statistical significance tests (paired t-test vs baselines)
- Segment analysis: accuracy by player type (power, contact, speed)
- Calibration metrics: reliability diagram data

**[MODIFY] `frontend/app/models/page.tsx`**
- Show cross-validation results
- Display per-target metric comparison table
- Add "Does Model Beat Baseline?" verdict with p-value

### Priority 10 — Production Hardening
**Impact: Low-Medium — needed for real deployment**

**[MODIFY] `backend/core/model_service.py`**
- Persist trained models to database (not just in-memory)
- Auto-reload latest model on startup
- Add model health check endpoint

**[NEW] `backend/tasks/daily_retrain.py`**
- Scheduled daily retraining with latest data
- Compare new model vs current, only deploy if better
- Log training results to `model_versions` table

**[MODIFY] `backend/main.py`**
- Add request logging middleware (prediction audit trail)
- Add rate limiting (e.g., 100 requests/min per IP)

**[NEW] `.github/workflows/ci.yml`** (if not exists)
- Run Python import checks
- Run frontend build
- Run any unit tests

---

## Quick Wins (< 1 day each)
- [ ] Add `StandardScaler` in `feature_builder.py` (Priority 2)
- [ ] Backfill `prediction_results` from completed games (Priority 4)
- [ ] Fix empty `/leaderboard` endpoint in `evaluation.py`
- [ ] Add CSV export button to predictions hub page
- [ ] Display existing `confidence` field on prediction cards in frontend
- [ ] Remove `version: "3.8"` from `docker-compose.yml` (deprecated warning)

---

## Implementation Order

```
Phase 2a (Week 1-2): Priorities 1-2 — Pitcher data + feature scaling
Phase 2b (Week 3-4): Priorities 3-4 — Game predictions + accuracy tracking
Phase 2c (Week 5-6): Priorities 5-6 — Advanced features + uncertainty
Phase 2d (Week 7-8): Priorities 7-9 — Model diversity + evaluation + pages
Phase 2e (Week 9):   Priority 10  — Production hardening
```

---

## Success Metrics
- **Model MSE drops >15%** after pitcher features + scaling (Priorities 1-2)
- **Game-level win probability** achieves >55% accuracy (Priority 3)
- **Prediction accuracy page** shows calibration within 10% (Priority 4)
- **Ensemble with 4+ models** outperforms best single model (Priority 7)
- **Cross-validation R²** consistently >0.15 for hits prediction (Priority 9)
