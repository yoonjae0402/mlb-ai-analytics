"""Pydantic request/response models for the API."""

from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel


# --- Train ---

class TrainRequest(BaseModel):
    epochs: int = 30
    lr: float = 0.001
    hidden_size: int = 64
    batch_size: int = 32
    n_estimators: int = 200
    max_depth: int = 6
    xgb_lr: float = 0.1
    seasons: list[int] = [2023, 2024]


class ModelMetrics(BaseModel):
    mse: float
    mae: float
    r2: float
    per_target: dict
    final_train_loss: float
    final_val_loss: float


class TrainResult(BaseModel):
    lstm: Optional[dict] = None
    xgboost: Optional[dict] = None


class TrainStatus(BaseModel):
    is_training: bool
    progress: float
    current_epoch: int
    total_epochs: int
    current_model: str
    train_loss: float
    val_loss: float


class TrainCurves(BaseModel):
    lstm: Optional[dict] = None
    xgboost: Optional[dict] = None


# --- Predict ---

class PredictRequest(BaseModel):
    player_id: int
    model_type: str = "lstm"  # "lstm", "xgboost", "ensemble"


class PredictionResponse(BaseModel):
    player_id: int
    model_type: str
    predicted_hits: float
    predicted_hr: float
    predicted_rbi: float
    predicted_walks: float
    all_predictions: dict
    feature_names: list[str]
    last_features: list[float]


# --- Players ---

class PlayerResponse(BaseModel):
    id: int
    mlb_id: int
    name: str
    team: Optional[str] = None
    position: Optional[str] = None
    bats: Optional[str] = None
    throws: Optional[str] = None
    headshot_url: Optional[str] = None
    current_level: Optional[str] = "MLB"
    prospect_rank: Optional[int] = None
    age: Optional[int] = None


class PlayerStatsResponse(BaseModel):
    game_date: date
    batting_avg: Optional[float] = None
    obp: Optional[float] = None
    slg: Optional[float] = None
    hits: Optional[int] = None
    home_runs: Optional[int] = None
    rbi: Optional[int] = None
    walks: Optional[int] = None


class PlayerDetail(BaseModel):
    player: PlayerResponse
    recent_stats: list[PlayerStatsResponse]
    season_totals: Optional[dict] = None


class PlayerIndexResponse(BaseModel):
    players: list[PlayerResponse]
    total: int
    page: int
    per_page: int


# --- Teams ---

class TeamResponse(BaseModel):
    id: int
    mlb_id: int
    name: str
    abbreviation: str
    league: Optional[str] = None
    division: Optional[str] = None
    logo_url: Optional[str] = None


# --- Attention ---

class AttentionRequest(BaseModel):
    sample_idx: int = 0


class AttentionResponse(BaseModel):
    attention_weights: list
    prediction: list[float]
    actual: list[float]
    feature_names: list[str]
    target_names: list[str]
    sample_idx: int
    n_samples: int


class FeatureAttributionResponse(BaseModel):
    feature_importance: list[float]
    feature_timestep_grads: list
    feature_names: list[str]
    sample_idx: int


# --- Ensemble ---

class EnsembleRequest(BaseModel):
    player_id: int
    strategy: str = "weighted_average"
    weights: Optional[list[float]] = None


class EnsembleResponse(BaseModel):
    player_id: int
    strategy: str
    weights: list[float]
    predictions: dict
    target_names: list[str]


class WeightSensitivityResponse(BaseModel):
    sweep: list[dict]


# --- Games ---

class GameResponse(BaseModel):
    game_id: str
    away_abbrev: str
    away_name: str
    home_abbrev: str
    home_name: str
    away_score: int
    home_score: int
    inning: int
    half: str
    status: str
    home_win_prob: float
    wp_history: list[float]
    home_probable_pitcher: Optional[str] = None
    away_probable_pitcher: Optional[str] = None
    game_datetime: Optional[str] = None
    venue: Optional[str] = None


class LiveGamesResponse(BaseModel):
    games: list[GameResponse]
    mode: str  # "live", "schedule", "off_day"


class ScheduleGameResponse(BaseModel):
    game_id: int
    game_date: str
    away_team: str
    home_team: str
    away_score: Optional[int] = None
    home_score: Optional[int] = None
    status: str
    venue: Optional[str] = None
    away_probable_pitcher: Optional[str] = None
    home_probable_pitcher: Optional[str] = None
    game_datetime: Optional[str] = None
    home_win_prob: Optional[float] = None


class ScheduleResponse(BaseModel):
    games: list[ScheduleGameResponse]
    start_date: str
    end_date: str


# --- Predictions Hub ---

class DailyPredictionResponse(BaseModel):
    player_id: int
    player_name: str
    team: Optional[str] = None
    headshot_url: Optional[str] = None
    predicted_hits: float
    predicted_hr: float
    predicted_rbi: float
    predicted_walks: float
    confidence: Optional[float] = None
    created_at: Optional[str] = None


class PredictionsHubResponse(BaseModel):
    predictions: list[DailyPredictionResponse]
    total: int
    last_updated: Optional[str] = None


# --- Evaluation ---

class EvaluationResponse(BaseModel):
    lstm: Optional[dict] = None
    xgboost: Optional[dict] = None
    baselines: Optional[dict] = None
    comparison: Optional[dict] = None


# --- Tuning ---

class TuneRequest(BaseModel):
    model_type: str = "lstm"  # "lstm" or "xgboost"
    n_trials: int = 50


class TuneStatus(BaseModel):
    is_tuning: bool
    model_type: str
    n_trials: int
    completed_trials: int
    best_params: Optional[dict] = None
    best_score: Optional[float] = None


# --- Data ---

class DataStatus(BaseModel):
    players_count: int
    stats_count: int
    games_count: int
    predictions_count: int
    last_updated: Optional[str] = None
    seasons: list[int]


# --- Game Detail & Predictions ---

class GameDetailResponse(BaseModel):
    game_id: int
    away_team: str
    home_team: str
    away_score: Optional[int] = None
    home_score: Optional[int] = None
    status: str
    venue: Optional[str] = None
    away_probable_pitcher: Optional[str] = None
    home_probable_pitcher: Optional[str] = None
    game_datetime: Optional[str] = None
    game_date: Optional[str] = None
    away_team_id: Optional[int] = None
    home_team_id: Optional[int] = None


class GamePlayerPrediction(BaseModel):
    player_id: int
    mlb_id: int
    name: str
    team: Optional[str] = None
    position: Optional[str] = None
    headshot_url: Optional[str] = None
    predicted_hits: Optional[float] = None
    predicted_hr: Optional[float] = None
    predicted_rbi: Optional[float] = None
    predicted_walks: Optional[float] = None
    confidence: Optional[float] = None
    has_prediction: bool = False


class GamePredictionsResponse(BaseModel):
    game: GameDetailResponse
    home_players: list[GamePlayerPrediction]
    away_players: list[GamePlayerPrediction]


# --- Player Compare ---

class PlayerCompareResponse(BaseModel):
    players: list[PlayerDetail]


# --- Scheduler ---

class SchedulerStatus(BaseModel):
    running: bool
    status: str
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    last_error: Optional[str] = None
