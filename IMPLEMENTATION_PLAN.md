# Goal Description
The goal is to ensure the MLB Analytics website shows player stats and allows searching even when no games are scheduled for the current day. This involves fixing a likely issue where the search component or data retrieval assumes active games. Additionally, the schedule component will be updated to be interactive, allowing users to click on a game to view detailed predictions for the players involved.

Furthermore, a significant design overhaul will be applied to make the site more "beginner friendly" (easier to read stats, professional color scheme, clear explanations), taking inspiration from a simplified FanGraphs aesthetic.

## User Review Required
None.

## Proposed Changes

### Frontend - Design Overhaul (Beginner Friendly)
- **Visual Style**: Switch to a cleaner, more professional color palette (modern slate/gray base instead of deep blue). Improve contrast and readability.
- **Context Badges**: Add "Elite" (Gold), "Great" (Green), "Average" (Gray), "Poor" (Orange) pills next to advanced stats.
- **Plain English Tooltips**: Replace math definitions with "Why it matters" explanations (e.g., "wRC+ measures total offense. 100 is average.").
- **Percentile Bars**: Visualize stats as 0-99 ratings (like video games) for instant understanding.
- **Trend Indicators**: Add simple ↑ ↓ arrows to show recent form (last 10 games vs season average).
- **Luck Meter**: Visual comparison of actual vs expected stats (e.g., wOBA vs xwOBA) to show if a player is "Lucky" or "Unlucky".

### Frontend - Features
- **Game Prediction Page**: Create `frontend/app/dashboard/game/[gameId]/page.tsx` using the new design system.
- **Update Schedule**: Add interaction to the schedule component to link to the new game prediction page.
- **Comparison Tool**: Create `frontend/app/dashboard/compare/page.tsx` for side-by-side player analysis. Use "Context Badges" to make advantages obvious (e.g., highlighting the text green if one player is "Elite" and the other "Average").
- **Search Robustness**: Verify player search handles empty states gracefully and ensure the issue isn't due to missing data.

### Backend
- **Get Game Endpoint**: Add `GET /v1/games/{game_id}` to `backend/api/v1/games.py` to fetch a specific game's details (using `statsapi.schedule`).
- **Game Predictions Endpoint**: Create `GET /v1/games/{game_id}/predictions`. Logic:
    1. Fetch game details to identify Home/Away teams.
    2. Fetch active rosters for both teams (cached via `pipeline.fetch_all_players` or similar).
    3. Query the `predictions` table for the latest prediction for each player on the roster.
    4. Return a structured response with game info and two lists of player predictions.

### Database / Data
- **Verify Seeding**: Ensure the `players` table is populated. The search issue might be due to an empty database if `seed_database` wasn't run or failed.

## Verification Plan
### Manual Verification
- **Visual Check**: Ensure new colors are professional and text is high-contrast.
- **Context Check**: Verify badges and tooltips appear correctly and match the data.
- **No Games Scenario**:
    - Verify player search works even if the schedule API returns nothing for today.
    - If the `games` table is empty in the DB, ensure search still queries the `players` table correctly.
- **Schedule Click**:
    - Click a game in the schedule view.
    - Verify navigation to `/dashboard/game/[gameId]`.
    - Verify the new page shows predictions for players on both teams.
- **Comparison Check**:
    - Select two players in the Comparison Tool.
    - Verify stats are shown side-by-side with correct "Elite/Avg" context badges.
