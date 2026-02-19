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

### Phase 2d — Model Diversity + Evaluation + Pages (Priorities 7-9) ✅
- [x] **LightGBM & Linear Models**: Added as alternatives to LSTM/XGBoost
- [x] **Ensemble**: Weighted averaging and stacking strategies
- [x] **Evaluation**: 5-fold CV, Wilcoxon tests, metrics (RMSE, R2, MAPE)
- [x] **Model Comparison UI**: Pages to compare models and visualize ensemble weights

### Phase 2e — Production Hardening (Priority 10) ✅
- [x] **Daily Retrain**: Champion/Challenger logic with `daily_retrain.py` (Fixed persistence issue)
- [x] **System Health**: Health check endpoint & dashboard
- [x] **Logging**: Middleware for request logging
- [x] **CI/CD**: GitHub workflows for testing and daily retraining

---

## Phase 3 — Real-World Value & Monetization (New)

### Phase 3a — Betting Intelligence & Odds
- [ ] **Odds API Integration**: Fetch live odds from major sportsbooks (DraftKings, FanDuel)
- [ ] **EV Calculation**: Compute Expected Value (EV) = (Model Prob * Decimal Odds) - 1
- [ ] **Value Dashboard**: `/dashboard/betting` showing high-EV bets
- [ ] **Bankroll Management**: Kelly Criterion suggestions based on confidence and bankroll size

### Phase 3b — User Personalization & Alerts
- [ ] **Authentication**: NextAuth.js (Google/GitHub login)
- [ ] **User Profile**: Saved preferences, bankroll settings
- [ ] **Favorites**: "Star" players/teams to track
- [ ] **Alert System**: Email/Push notifications when:
    -   A "Value Bet" > 10% EV is found
    -   A favorite player is predicted to hit a HR
    -   A favorite team's win probability shifts significantly

### Phase 3c — Advanced Sabermetrics & Refinement
- [ ] **Advanced Pitching**: FIP (Fielding Independent Pitching), xFIP, K-BB%
- [ ] **Advanced Batting**: wRC+ (Weighted Runs Created Plus), ISO (Isolated Power)
- [ ] **WAR Proxy**: Simplified Wins Above Replacement calculation
- [ ] **Statcast Integration**: Visualize Exit Velo / Launch Angle stability over time
- [ ] **True Time-Series CV**: Refactor evaluation to retrain models on expanding windows (fix current data leakage)


---

## Implementation Order

```
Phase 2d ✅:  Priorities 7-9 — Model diversity + evaluation + pages
Phase 2e ✅:  Priority 10    — Production hardening
Phase 3a   :  Betting Intelligence (Highest Value)
Phase 3b   :  User Personalization
Phase 3c   :  Advanced Sabermetrics
```

## Key Rule
> **Every backend feature MUST have a corresponding frontend page or component.**
> Never add an API endpoint without wiring it into the UI so users can actually see and use it.

---

## Success Metrics
- **Model MSE drops >15%** after pitcher features + scaling (DONE - Phase 2a)
- **Game-level win probability** achieves >55% accuracy (DONE - Phase 2b)
- **Ensemble with 4+ models** outperforms best single model (DONE - Phase 2d)
- **Positive EV Returns**: Simulated betting portfolio yields >5% ROI (Phase 3a)
- **User Engagement**: value-added features like Alerts drive daily active usage (Phase 3b)

