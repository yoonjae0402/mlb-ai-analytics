# MLB AI Analytics — Implementation Plan

## Completed

### Phase 1 — Docker Auto-Seeding + Baseline Predictions
- [x] Docker entrypoint auto-seeds MLB + AAA data on first startup
- [x] Baseline predictions via weighted historical averages (no ML needed)
- [x] Full Docker Compose stack: PostgreSQL + FastAPI + Next.js
- [x] Code quality cleanup (unused imports in predict.py, train.py, games.py)

### Phase 2a — Pitcher Matchups + Feature Scaling
- [x] **Feature scaling**: StandardScaler in `feature_builder.py` (fit on train only, persist to disk)
- [x] **22 features**: expanded from 15 (added pitcher matchup + derived features)
- [x] **Pitcher ingestion**: `pipeline.py` fetches pitching game logs, stores IP + earned runs
- [x] **Pitcher API**: `GET /v1/pitchers/search`, `GET /v1/pitchers/{id}/stats`
- [x] **Pitcher DB fields**: `innings_pitched`, `earned_runs` on PlayerStat
- [x] **Leaderboard fix**: real ranked predictions at `GET /v1/leaderboard`
- [x] **docker-compose**: removed deprecated `version: "3.8"`

### Phase 2b — Game Win Probability + Prediction Accuracy
- [x] **Win probability**: `game_predictor.py` — Pythagorean expectation from player predictions
- [x] **Win prob API**: `GET /v1/games/{game_id}/win-probability`
- [x] **Win prob UI**: probability bar + team projections on game detail page
- [x] **Backfill task**: `backfill_results.py` — compares predictions vs actuals
- [x] **Accuracy API**: `/v1/accuracy/summary`, `/by-player/{id}`, `/calibration`, `POST /backfill`
- [x] **Accuracy dashboard**: `/dashboard/accuracy` with metrics, per-stat breakdown, calibration chart
- [x] **Leaderboard page**: `/dashboard/leaderboard` with ranked players + composite scores
- [x] **Navigation**: sidebar + home page updated with Leaderboard, Accuracy links
- [x] **Pitcher stats page**: `/dashboard/pitchers` with search and stats display
- [x] **Frontend parity audit**: all backend features exposed in UI, confidence scores displayed

### Phase 2c — Advanced Feature Engineering + Uncertainty
- [x] **26 features**: expanded from 22 with BABIP, cold streak, home/away, opponent quality
- [x] **BABIP**: `(H - HR) / (AB - K - HR)` — luck vs skill separation
- [x] **Cold streak**: 1 if BA < .150 last 7 games — slump detection
- [x] **Home/away**: binary from Game table lookup — split performance
- [x] **Opponent quality**: team winning % from standings — schedule strength
- [x] **Game lookup**: `_build_game_lookup()` for home/away derivation
- [x] **Team quality lookup**: `_build_team_quality()` for opponent win %
- [x] **Monte Carlo Dropout**: `predict_with_uncertainty()` — 30 forward passes, 90% CI
- [x] **API schema**: `confidence_interval_low`, `confidence_interval_high`, `uncertainty` on PredictionResponse
- [x] **CI visualization**: color-coded confidence interval bars on predict page (green/yellow/red)
- [x] **Architecture page**: updated to 26 features, (batch, 10, 26), MC Dropout documented
- [x] **Radar chart**: updated with BABIP, opp_quality in key features

---

## Remaining Work

### Phase 2d — Model Diversity + Evaluation + Pages (Priorities 7-9)

#### Priority 7 — Model Diversity & Ensemble ✅
**Impact: Medium — more diverse models = better ensemble**

**Backend:**
- [x] `src/models/lightgbm_model.py` — LightGBM regressor with same feature flattening as XGBoost
- [x] `src/models/linear_model.py` — Ridge/Lasso regression baseline
- [x] `src/models/ensemble.py` — Support 3+ base models, `optimize_weights()` via grid search
- [x] `src/models/model_registry.py` — `train_lightgbm()` and `train_linear()` functions
- [x] `backend/core/model_service.py` — lightgbm/linear support, N-model ensemble, `get_trained_model_names()`
- [x] `backend/api/v1/train.py` — saves lightgbm/linear to DB
- [x] `backend/api/v1/schemas.py` — TrainRequest: `train_lightgbm`, `train_linear` flags; TrainResult/TrainCurves: all 4 models
- [x] `backend/api/v1/evaluation.py` — returns lightgbm/linear metrics
- [x] `backend/requirements.txt` — lightgbm>=4.1.0

**Frontend:**
- [x] `frontend/app/models/page.tsx` — Model comparison table (LSTM, XGBoost, LightGBM, Linear), BEST badge, checkboxes to enable extra models
- [x] `frontend/app/ensemble/page.tsx` — Per-model weight sliders, normalized blend bar, N-model support
- [x] `frontend/components/charts/BarComparison.tsx` — Dynamic keys (any model set)
- [x] `frontend/lib/api.ts` — TrainConfig with lightgbm/linear fields

#### Priority 8 — Missing Frontend Pages ✅
**Impact: Medium — completes the product**

**Frontend:**
- [x] `frontend/app/dashboard/player/[playerId]/page.tsx` — Player detail: career stats table, prediction history line chart, season totals, sparkline stat averages
- [x] `frontend/app/tuning/page.tsx` — Start/monitor Optuna HPT: model selector, trial count, progress bar, best params display
- [x] `frontend/app/dashboard/predictions/page.tsx` — CSV export button (client-side, date-stamped filename)
- [x] `frontend/lib/constants.ts` — Added "Hyperparameter Tuning" to Analysis nav section
- [x] `frontend/components/layout/ModernSidebar.tsx` — Added Sliders icon
- [x] `frontend/app/dashboard/players/page.tsx` — Player names now link to `/dashboard/player/[id]`

#### Priority 9 — Evaluation Rigor ✅
**Impact: Medium — proves analytical credibility**

**Backend:**
- [x] `backend/core/evaluation.py` — 5-fold time-series CV (expanding window), RMSE/MAPE/R² per target, Wilcoxon signed-rank significance tests vs baselines, bootstrap 95% CI

**Frontend:**
- [x] `frontend/app/models/page.tsx` — CV results table (mean/std MSE per fold), "Does Model Beat Baseline?" verdict table with p-value, Beats/Marginal/Loses badges
- [x] `frontend/lib/api.ts` — EvaluationResult extended with cv_results, lightgbm, linear fields

---

### Phase 2e — Production Hardening (Priority 10)

**Backend:**
- [MODIFY] `backend/core/model_service.py` — Persist trained models to database (not just in-memory), auto-reload latest on startup, model health check endpoint
- [NEW] `backend/tasks/daily_retrain.py` — Scheduled retraining, compare new vs current, only deploy if better
- [MODIFY] `backend/main.py` — Request logging middleware, rate limiting

**Frontend:**
- [NEW] `frontend/app/dashboard/system/page.tsx` — System health dashboard: model status, DB stats, API latency, last retrain timestamp
- [MODIFY] `frontend/lib/constants.ts` — Add System Health to nav

**DevOps:**
- [NEW] `.github/workflows/ci.yml` — Python import checks, frontend build, unit tests

---

## Implementation Order

```
Phase 2d (Next):    Priorities 7-9 — Model diversity + evaluation + pages
Phase 2e (Final):   Priority 10    — Production hardening
```

## Key Rule
> **Every backend feature MUST have a corresponding frontend page or component.**
> Never add an API endpoint without wiring it into the UI so users can actually see and use it.

---

## Success Metrics
- **Model MSE drops >15%** after pitcher features + scaling (DONE - Phase 2a)
- **Game-level win probability** achieves >55% accuracy (DONE - Phase 2b)
- **Prediction accuracy page** shows calibration within 10% (DONE - Phase 2b)
- **Uncertainty intervals** cover actual values >80% of the time (DONE - Phase 2c)
- **Ensemble with 4+ models** outperforms best single model (Phase 2d)
- **Cross-validation R²** consistently >0.15 for hits prediction (Phase 2d)
