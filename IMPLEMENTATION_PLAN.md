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
