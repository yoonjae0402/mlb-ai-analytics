    # Goal Description
The goal is to ensure the MLB Analytics website shows player stats and allows searching even when no games are scheduled for the current day. This involves fixing a likely issue where the search component or data retrieval assumes active games. Additionally, the schedule component will be updated to be interactive, allowing users to click on a game to view detailed predictions for the players involved.

Furthermore, a significant design overhaul will be applied to make the site more "beginner friendly" (easier to read stats, professional color scheme, clear explanations), taking inspiration from a simplified FanGraphs aesthetic.

## User Review Required
None.

## Proposed Changes

### Frontend - Design Overhaul & Integration
- **Player Card (`frontend/components/cards/PlayerCard.tsx`)**:
    - Add `ContextBadge` to show player tier (e.g., "Elite" badge next to name).
    - Use `TrendIndicator` for recent form.
- **Player Stats**:
    - Update `StatGauge.tsx` to use the new `ContextBadge` logic and colors.
    - Add `InfoTooltip` to stat labels in all dashboards.
- **Integration**:
    - Ensure `lib/stat-helpers.ts` is the single source of truth for thresholds and colors.

### Frontend - Features
- **Game Prediction Page**: Create `frontend/app/dashboard/game/[gameId]/page.tsx` using the new design system.
- **Update Schedule**: Add interaction to the schedule component to link to the new game prediction page.
- **Comparison Tool**: Create `frontend/app/dashboard/compare/page.tsx` for side-by-side player analysis. Use "Context Badges" to make advantages obvious (e.g., highlighting the text green if one player is "Elite" and the other "Average").
- **Search Robustness**: Verify player search handles empty states gracefully and ensure the issue isn't due to missing data.

### Backend
- **Get Game Endpoint**: Verify `GET /v1/games/{game_id}` (ALREADY IMPLEMENTED).
- **Game Predictions Endpoint**: Verify `GET /v1/games/{game_id}/predictions` (ALREADY IMPLEMENTED).
- **Action**: No new backend code needed for these features. Focus on testing they return correct data.

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
