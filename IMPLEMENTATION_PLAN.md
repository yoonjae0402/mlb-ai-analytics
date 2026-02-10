# Implementation Plan - "Pro Analytics" Dashboard & MiLB Expansion

Goal: Transform the project into a professional-grade MLB analytics platform (similar to Baseball Savant/FanGraphs) with full minor league coverage and rich visual assets.

## User Review Required
> [!IMPORTANT]
> This is a major scope expansion. Database size will increase significantly with MiLB players.
> I will need to use standard MLB image endpoints. If these change or have strict CORS policies, images might not load directly on the frontend without a proxy.

## Proposed Changes

### 1. Data Pipeline & Automation (No Manual Training)
#### [NEW] [src/services/scheduler.py]
- **Automated Prediction Job**: Cron job (e.g., 4 AM ET) that:
    1. Fetches latest game data.
    2. Retrains/Fine-tunes models on new data.
    3. Generates predictions for *all* upcoming games.
    4. Caches results in the DB for instant frontend access.

#### [MODIFY] [pipeline.py](file:///Users/yunjaejung/Desktop/mlb-ai-analytics/src/data/pipeline.py)
- **Full Roster Fetch**: Add `fetch_all_players(season)` to get everyone on 40-man rosters + top prospects.
- **MiLB Stats**: Update `fetch_batting_stats` to include minor league levels (AAA, AA, etc.).
- **Image URLs**: Add helper methods to generate headshot and logo URLs based on IDs.

#### [MODIFY] [models.py](file:///Users/yunjaejung/Desktop/mlb-ai-analytics/backend/db/models.py)
- **Enhanced Player Model**: Add fields for `headshot_url`, `current_level` (MLB/AAA/etc.), `prospect_rank`.
- **Team Model**: Add `logo_url`, `abbreviation`, `league`, `division`.
- **Optimization**:
    - **Composite Indexes**: Add indexes on `(team, current_level)` for fast filtering.
    - **Date Partitioning** (Future Proofing): Prepare `player_stats` for partitioning by season if rows exceed 10M.
    - **Data Types**: Use `SmallInteger` for count stats (hits, HRs) to save space.

### 2. Frontend Overhaul (Professional UI)
#### [NEW] [components/layout/ModernSidebar.tsx]
- Create a collapsible, professional sidebar navigation (Scores/Schedule, Predictions, Players, Teams, Leaders, Analysis).

#### [NEW] [components/visuals/PlayerHeadshot.tsx]
- Component to handle player images with fallbacks for missing photos (common in MiLB).

#### [NEW] [components/visuals/TeamLogo.tsx]
- SVG logo component.

#### [MODIFY] [app/dashboard/page.tsx]
- **Dashboard Redesign**: Replace "Portfolio" look with a dense data dashboard.
- **Remove**: "Train Model" controls.
- **Add**: "System Status" widget showing "Last Updated: [Time]" and "Next Prediction: [Time]".
- **Widgets**:
    - **"Benefit of the Doubt"**: Top 5 predictions where our model disagrees with Vegas odds.

#### [NEW] [app/dashboard/players/page.tsx]
- **Player Index**: Searchable, filterable table of ALL players (MLB + MiLB).
- **Filters**: Team, Level, Position, Age.

#### [NEW] [app/dashboard/schedule/page.tsx]
- **Calendar View**: Monthly/Weekly view of games.
- **Game Cards**: Show probable pitchers, time, and *win probability* (if game is live/future).

#### [NEW] [app/dashboard/predictions/page.tsx]
- **Prediction Hub**: Central place for all model outputs.
- **Daily Best Bets**: Top high-confidence predictions for today's games.

### 3. ML: "Call-Up" Analysis
#### [NEW] [src/analysis/prospect_ranking.py]
- **Translation Factors**: Implement Minor League layout translation (MLE) logic to project MiLB stats to MLB equivalents.
- **Projection**: Use the existing LSTM model on *translated* MiLB stats to predict MLB impact.

## Verification Plan

### Automated Tests
- Verify `statsapi` returns MiLB data correctly.
- Test "Translation Factors" logic with known players (e.g., Jackson Holliday's 2024 projections).

### Manual Verification
- **Visual Check**: Ensure team logos and player headshots load correctly on the dashboard.
- **Data Completeness**: Search for a known minor leaguer (e.g., a top 100 prospect) and verify their stats appear.
- **UI UX**: Confirm the "Portfolio" feel is gone, replaced by a dense, data-rich interface.
