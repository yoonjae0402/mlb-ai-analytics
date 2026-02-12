"""Live games, game detail, and game predictions endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, or_

from backend.api.v1.schemas import (
    LiveGamesResponse, GameResponse, GameDetailResponse,
    GamePlayerPrediction, GamePredictionsResponse,
    WinProbabilityResponse, TeamProjectionResponse,
)
from backend.db.session import get_db
from backend.db.models import Player, Prediction
from src.services.realtime import fetch_live_games

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/games", tags=["games"])


@router.get("/live", response_model=LiveGamesResponse)
async def live_games():
    """Get live or scheduled games from MLB Stats API."""
    games, mode = fetch_live_games()
    return LiveGamesResponse(
        games=[GameResponse(**g) for g in games],
        mode=mode,
    )


@router.get("/today", response_model=LiveGamesResponse)
async def todays_games():
    """Get today's schedule with predictions."""
    games, mode = fetch_live_games()
    return LiveGamesResponse(
        games=[GameResponse(**g) for g in games],
        mode=mode,
    )


@router.get("/{game_id}", response_model=GameDetailResponse)
async def get_game_detail(game_id: int):
    """Get details for a specific game."""
    try:
        import statsapi
        schedule = statsapi.schedule(game_id=game_id)
    except Exception as e:
        logger.error(f"Failed to fetch game {game_id}: {e}")
        raise HTTPException(status_code=502, detail=f"MLB API error: {e}")

    if not schedule:
        raise HTTPException(status_code=404, detail="Game not found")

    g = schedule[0]
    return GameDetailResponse(
        game_id=g.get("game_id", game_id),
        away_team=g.get("away_name", ""),
        home_team=g.get("home_name", ""),
        away_score=g.get("away_score"),
        home_score=g.get("home_score"),
        status=g.get("status", "Scheduled"),
        venue=g.get("venue_name", ""),
        away_probable_pitcher=g.get("away_probable_pitcher", "TBD"),
        home_probable_pitcher=g.get("home_probable_pitcher", "TBD"),
        game_datetime=g.get("game_datetime", ""),
        game_date=g.get("game_date", ""),
        away_team_id=g.get("away_id"),
        home_team_id=g.get("home_id"),
    )


@router.get("/{game_id}/predictions", response_model=GamePredictionsResponse)
async def get_game_predictions(
    game_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get player predictions for a specific game.

    Fetches rosters for both teams and their latest predictions.
    """
    # 1. Get game details
    try:
        import statsapi
        schedule = statsapi.schedule(game_id=game_id)
    except Exception as e:
        logger.error(f"Failed to fetch game {game_id}: {e}")
        raise HTTPException(status_code=502, detail=f"MLB API error: {e}")

    if not schedule:
        raise HTTPException(status_code=404, detail="Game not found")

    g = schedule[0]
    game_detail = GameDetailResponse(
        game_id=g.get("game_id", game_id),
        away_team=g.get("away_name", ""),
        home_team=g.get("home_name", ""),
        away_score=g.get("away_score"),
        home_score=g.get("home_score"),
        status=g.get("status", "Scheduled"),
        venue=g.get("venue_name", ""),
        away_probable_pitcher=g.get("away_probable_pitcher", "TBD"),
        home_probable_pitcher=g.get("home_probable_pitcher", "TBD"),
        game_datetime=g.get("game_datetime", ""),
        game_date=g.get("game_date", ""),
        away_team_id=g.get("away_id"),
        home_team_id=g.get("home_id"),
    )

    # 2. Get rosters - match players from our DB by team ID -> abbreviation
    home_id = g.get("home_id")
    away_id = g.get("away_id")
    home_team_name = g.get("home_name", "")
    away_team_name = g.get("away_name", "")

    async def get_team_players(mlb_team_id: int, team_name_fallback: str) -> list[GamePlayerPrediction]:
        """Look up players in our DB for this team and get their latest predictions."""
        # Resolve team abbreviation from DB if possible
        from backend.db.models import Team

        search_term = team_name_fallback
        
        if mlb_team_id:
            team_q = await db.execute(select(Team).where(Team.mlb_id == mlb_team_id))
            team_obj = team_q.scalars().one_or_none()
            if team_obj and team_obj.abbreviation:
                search_term = team_obj.abbreviation
        
        # Query players matching abbreviation OR full name (just in case)
        result = await db.execute(
            select(Player)
            .where(
                or_(
                    Player.team == search_term,  # Exact match for abbreviation (e.g. "NYY")
                    Player.team.ilike(f"%{team_name_fallback}%"), # Fallback for full name
                    Player.team.ilike(f"%{search_term}%") # Partial match for abbreviation
                )
            )
            .order_by(Player.name)
        )
        players = result.scalars().all()

        player_preds = []
        for p in players:
            # Get latest prediction for this player
            pred_result = await db.execute(
                select(Prediction)
                .where(Prediction.player_id == p.id)
                .order_by(desc(Prediction.created_at))
                .limit(1)
            )
            pred = pred_result.scalar_one_or_none()

            # If no prediction exists, generate a baseline one on-the-fly
            pred_hits = None
            pred_hr = None
            pred_rbi = None
            pred_walks = None
            confidence = None
            has_pred = False

            if pred:
                pred_hits = pred.predicted_hits
                pred_hr = pred.predicted_hr
                pred_rbi = pred.predicted_rbi
                pred_walks = pred.predicted_walks
                confidence = pred.confidence
                has_pred = True
            else:
                # Auto-generate baseline prediction from historical stats
                try:
                    from backend.services.baseline_predictor import generate_player_prediction, _ensure_baseline_model_version
                    from backend.db.session import SyncSessionLocal
                    
                    sync_session = SyncSessionLocal()
                    try:
                        sync_player = sync_session.query(Player).filter_by(id=p.id).first()
                        if sync_player:
                            mv = _ensure_baseline_model_version(sync_session)
                            baseline = generate_player_prediction(sync_session, sync_player, mv)
                            if baseline:
                                pred_hits = baseline["predicted_hits"]
                                pred_hr = baseline["predicted_hr"]
                                pred_rbi = baseline["predicted_rbi"]
                                pred_walks = baseline["predicted_walks"]
                                confidence = baseline["confidence"]
                                has_pred = True
                                
                                # Store it for future requests
                                from backend.db.models import Prediction as PredModel
                                new_pred = PredModel(
                                    player_id=p.id,
                                    model_version_id=mv.id,
                                    predicted_hits=pred_hits,
                                    predicted_hr=pred_hr,
                                    predicted_rbi=pred_rbi,
                                    predicted_walks=pred_walks,
                                    confidence=confidence,
                                )
                                sync_session.add(new_pred)
                                sync_session.commit()
                    finally:
                        sync_session.close()
                except Exception as e:
                    logger.warning(f"Could not generate baseline prediction for {p.name}: {e}")

            player_preds.append(GamePlayerPrediction(
                player_id=p.id,
                mlb_id=p.mlb_id,
                name=p.name,
                team=p.team,
                position=p.position,
                headshot_url=p.headshot_url,
                predicted_hits=pred_hits,
                predicted_hr=pred_hr,
                predicted_rbi=pred_rbi,
                predicted_walks=pred_walks,
                confidence=confidence,
                has_prediction=has_pred,
            ))

        return player_preds

    home_players = await get_team_players(home_id, home_team_name)
    away_players = await get_team_players(away_id, away_team_name)

    return GamePredictionsResponse(
        game=game_detail,
        home_players=home_players,
        away_players=away_players,
    )


@router.get("/{game_id}/win-probability", response_model=WinProbabilityResponse)
async def get_win_probability(game_id: int):
    """Compute win probability for a game based on player predictions.

    Aggregates individual player predictions into projected team runs,
    then applies Pythagorean expectation to derive win probability.
    """
    try:
        import statsapi
        schedule = statsapi.schedule(game_id=game_id)
    except Exception as e:
        logger.error(f"Failed to fetch game {game_id}: {e}")
        raise HTTPException(status_code=502, detail=f"MLB API error: {e}")

    if not schedule:
        raise HTTPException(status_code=404, detail="Game not found")

    g = schedule[0]

    from backend.db.session import SyncSessionLocal
    from backend.services.game_predictor import compute_win_probability

    session = SyncSessionLocal()
    try:
        result = compute_win_probability(
            session,
            home_team_id=g.get("home_id", 0),
            away_team_id=g.get("away_id", 0),
            home_team_name=g.get("home_name", ""),
            away_team_name=g.get("away_name", ""),
        )

        return WinProbabilityResponse(
            home_win_pct=result.home_win_pct,
            away_win_pct=result.away_win_pct,
            home=TeamProjectionResponse(
                team_name=result.home.team_name,
                team_abbreviation=result.home.team_abbreviation,
                projected_runs=result.home.projected_runs,
                projected_hits=result.home.projected_hits,
                projected_hr=result.home.projected_hr,
                projected_rbi=result.home.projected_rbi,
                projected_walks=result.home.projected_walks,
                n_players_with_predictions=result.home.n_players_with_predictions,
                n_total_players=result.home.n_total_players,
            ),
            away=TeamProjectionResponse(
                team_name=result.away.team_name,
                team_abbreviation=result.away.team_abbreviation,
                projected_runs=result.away.projected_runs,
                projected_hits=result.away.projected_hits,
                projected_hr=result.away.projected_hr,
                projected_rbi=result.away.projected_rbi,
                projected_walks=result.away.projected_walks,
                n_players_with_predictions=result.away.n_players_with_predictions,
                n_total_players=result.away.n_total_players,
            ),
            confidence=result.confidence,
            method=result.method,
        )
    finally:
        session.close()
