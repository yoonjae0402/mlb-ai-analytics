const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }
  return res.json();
}

// Health
export const getHealth = () => fetchAPI<{ status: string }>("/health");

// Training
export const startTraining = (config: TrainConfig) =>
  fetchAPI<{ status: string }>("/v1/train", {
    method: "POST",
    body: JSON.stringify(config),
  });

export const getTrainStatus = () => fetchAPI<TrainStatus>("/v1/train/status");
export const getTrainCurves = () => fetchAPI<TrainCurves>("/v1/train/curves");

// Players
export const searchPlayers = (q: string) =>
  fetchAPI<Player[]>(`/v1/players/search?q=${encodeURIComponent(q)}`);

export const getPlayer = (id: number) =>
  fetchAPI<PlayerDetail>(`/v1/players/${id}`);

export const getPlayerPredictions = (id: number) =>
  fetchAPI<PredictionRecord[]>(`/v1/players/${id}/predictions`);

export const getPlayerIndex = (params: {
  page?: number;
  per_page?: number;
  team?: string;
  level?: string;
  position?: string;
  search?: string;
}) => {
  const qs = new URLSearchParams();
  if (params.page) qs.set("page", String(params.page));
  if (params.per_page) qs.set("per_page", String(params.per_page));
  if (params.team) qs.set("team", params.team);
  if (params.level) qs.set("level", params.level);
  if (params.position) qs.set("position", params.position);
  if (params.search) qs.set("search", params.search);
  return fetchAPI<PlayerIndexResult>(`/v1/players/index?${qs.toString()}`);
};

// Teams
export const getTeams = () => fetchAPI<TeamData[]>("/v1/teams/");

// Predictions
export const predictPlayer = (playerId: number, modelType = "lstm") =>
  fetchAPI<PredictionResult>("/v1/predict/player", {
    method: "POST",
    body: JSON.stringify({ player_id: playerId, model_type: modelType }),
  });

// Predictions Hub
export const getDailyPredictions = (sortBy = "predicted_hr", limit = 50) =>
  fetchAPI<PredictionsHubResult>(
    `/v1/predictions/daily?sort_by=${sortBy}&limit=${limit}`
  );

export const getBestBets = (limit = 5) =>
  fetchAPI<PredictionsHubResult>(`/v1/predictions/best-bets?limit=${limit}`);

// Leaderboard
export const getLeaderboard = (limit = 25) =>
  fetchAPI<LeaderboardEntry[]>(`/v1/leaderboard?limit=${limit}`);

// Pitchers
export const searchPitchers = (q = "", limit = 20) =>
  fetchAPI<Player[]>(`/v1/pitchers/search?q=${encodeURIComponent(q)}&limit=${limit}`);

export const getPitcherStats = (pitcherId: number) =>
  fetchAPI<PitcherStatsResult>(`/v1/pitchers/${pitcherId}/stats`);

// Attention
export const getAttentionWeights = (sampleIdx = 0) =>
  fetchAPI<AttentionResult>("/v1/attention/weights", {
    method: "POST",
    body: JSON.stringify({ sample_idx: sampleIdx }),
  });

export const getFeatureAttribution = (sampleIdx = 0) =>
  fetchAPI<FeatureAttributionResult>("/v1/attention/feature-attribution", {
    method: "POST",
    body: JSON.stringify({ sample_idx: sampleIdx }),
  });

// Ensemble
export const getEnsemblePrediction = (
  playerId: number,
  strategy = "weighted_average",
  weights?: number[]
) =>
  fetchAPI<EnsembleResult>("/v1/ensemble/predict", {
    method: "POST",
    body: JSON.stringify({ player_id: playerId, strategy, weights }),
  });

export const getWeightSensitivity = () =>
  fetchAPI<WeightSensitivityResult>("/v1/ensemble/weight-sensitivity");

// Games
export const getLiveGames = () => fetchAPI<LiveGamesResult>("/v1/games/live");
export const getTodayGames = () => fetchAPI<LiveGamesResult>("/v1/games/today");
export const getGameDetail = (gameId: number | string) =>
  fetchAPI<GameDetail>(`/v1/games/${gameId}`);
export const getGamePredictions = (gameId: number | string) =>
  fetchAPI<GamePredictionsResult>(`/v1/games/${gameId}/predictions`);

// Player Compare
export const comparePlayers = (ids: number[]) =>
  fetchAPI<PlayerCompareResult>(`/v1/players/compare?ids=${ids.join(",")}`);

// Schedule
export const getScheduleRange = (startDate?: string, endDate?: string) => {
  const qs = new URLSearchParams();
  if (startDate) qs.set("start_date", startDate);
  if (endDate) qs.set("end_date", endDate);
  return fetchAPI<ScheduleResult>(`/v1/schedule/range?${qs.toString()}`);
};

export const getScheduleToday = () =>
  fetchAPI<ScheduleResult>("/v1/schedule/today");

// Model metrics
export const getModelMetrics = () => fetchAPI<Record<string, any>>("/v1/model/metrics");
export const getModelEvaluation = () => fetchAPI<EvaluationResult>("/v1/model/evaluation");

// Data
export const getDataStatus = () => fetchAPI<DataStatus>("/v1/data/status");
export const refreshData = () =>
  fetchAPI<{ status: string }>("/v1/data/refresh", { method: "POST" });

// Tuning
export const startTuning = (modelType: string, nTrials = 50) =>
  fetchAPI<{ status: string }>("/v1/tune", {
    method: "POST",
    body: JSON.stringify({ model_type: modelType, n_trials: nTrials }),
  });
export const getTuningStatus = () => fetchAPI<TuneStatus>("/v1/tune/status");

// Scheduler
export const getSchedulerStatus = () =>
  fetchAPI<SchedulerStatusResult>("/v1/scheduler/status");

export const triggerSchedulerRun = () =>
  fetchAPI<{ status: string }>("/v1/scheduler/run", { method: "POST" });

// Types
export interface TrainConfig {
  epochs?: number;
  lr?: number;
  hidden_size?: number;
  batch_size?: number;
  n_estimators?: number;
  max_depth?: number;
  xgb_lr?: number;
  seasons?: number[];
  train_lightgbm?: boolean;
  train_linear?: boolean;
  num_leaves?: number;
  linear_alpha?: number;
  linear_model_type?: string;
}

export interface TrainStatus {
  is_training: boolean;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  current_model: string;
  train_loss: number;
  val_loss: number;
}

export interface TrainCurves {
  lstm?: { train_losses: number[]; val_losses: number[] };
  xgboost?: { train_losses: number[]; val_losses: number[] };
}

export interface Player {
  id: number;
  mlb_id: number;
  name: string;
  team?: string;
  position?: string;
  bats?: string;
  throws?: string;
  headshot_url?: string;
  current_level?: string;
  prospect_rank?: number;
  age?: number;
}

export interface PlayerStat {
  game_date: string;
  batting_avg?: number;
  obp?: number;
  slg?: number;
  hits?: number;
  home_runs?: number;
  rbi?: number;
  walks?: number;
}

export interface PlayerDetail {
  player: Player;
  recent_stats: PlayerStat[];
  season_totals?: Record<string, number>;
}

export interface PlayerIndexResult {
  players: Player[];
  total: number;
  page: number;
  per_page: number;
}

export interface TeamData {
  id: number;
  mlb_id: number;
  name: string;
  abbreviation: string;
  league?: string;
  division?: string;
  logo_url?: string;
}

export interface PredictionResult {
  player_id: number;
  model_type: string;
  predicted_hits: number;
  predicted_hr: number;
  predicted_rbi: number;
  predicted_walks: number;
  all_predictions: Record<string, number[]>;
  feature_names: string[];
  last_features: number[];
  confidence_interval_low?: number[];
  confidence_interval_high?: number[];
  uncertainty?: number[];
}

export interface PredictionRecord {
  id: number;
  predicted_hits: number;
  predicted_hr: number;
  predicted_rbi: number;
  predicted_walks: number;
  created_at: string;
}

export interface DailyPrediction {
  player_id: number;
  player_name: string;
  team?: string;
  headshot_url?: string;
  predicted_hits: number;
  predicted_hr: number;
  predicted_rbi: number;
  predicted_walks: number;
  confidence?: number;
  created_at?: string;
}

export interface PredictionsHubResult {
  predictions: DailyPrediction[];
  total: number;
  last_updated?: string;
}

export interface AttentionResult {
  attention_weights: number[][][];
  prediction: number[];
  actual: number[];
  feature_names: string[];
  target_names: string[];
  sample_idx: number;
  n_samples: number;
}

export interface FeatureAttributionResult {
  feature_importance: number[];
  feature_timestep_grads: number[][];
  feature_names: string[];
  sample_idx: number;
}

export interface EnsembleResult {
  player_id: number;
  strategy: string;
  weights: number[];
  predictions: Record<string, number[]>;
  target_names: string[];
}

export interface WeightSensitivityResult {
  sweep: { lstm_weight: number; mse: number }[];
}

export interface GameData {
  game_id: string;
  away_abbrev: string;
  away_name: string;
  home_abbrev: string;
  home_name: string;
  away_score: number;
  home_score: number;
  inning: number;
  half: string;
  status: string;
  home_win_prob: number;
  wp_history: number[];
  home_probable_pitcher?: string;
  away_probable_pitcher?: string;
  game_datetime?: string;
  venue?: string;
}

export interface LiveGamesResult {
  games: GameData[];
  mode: string;
}

export interface ScheduleGame {
  game_id: number;
  game_date: string;
  away_team: string;
  home_team: string;
  away_score?: number;
  home_score?: number;
  status: string;
  venue?: string;
  away_probable_pitcher?: string;
  home_probable_pitcher?: string;
  game_datetime?: string;
  home_win_prob?: number;
}

export interface ScheduleResult {
  games: ScheduleGame[];
  start_date: string;
  end_date: string;
}

export interface DataStatus {
  players_count: number;
  stats_count: number;
  games_count: number;
  predictions_count: number;
  last_updated?: string;
  seasons: number[];
}

export interface EvaluationResult {
  lstm?: Record<string, any>;
  xgboost?: Record<string, any>;
  baselines?: Record<string, any>;
  comparison?: Record<string, any>;
}

export interface TuneStatus {
  is_tuning: boolean;
  model_type: string;
  n_trials: number;
  completed_trials: number;
  best_params?: Record<string, any>;
  best_score?: number;
}

export interface SchedulerStatusResult {
  running: boolean;
  status: string;
  last_run?: string;
  next_run?: string;
  last_error?: string;
}

// Game Detail & Predictions
export interface GameDetail {
  game_id: number;
  away_team: string;
  home_team: string;
  away_score?: number;
  home_score?: number;
  status: string;
  venue?: string;
  away_probable_pitcher?: string;
  home_probable_pitcher?: string;
  game_datetime?: string;
  game_date?: string;
  away_team_id?: number;
  home_team_id?: number;
}

export interface GamePlayerPrediction {
  player_id: number;
  mlb_id: number;
  name: string;
  team?: string;
  position?: string;
  headshot_url?: string;
  predicted_hits?: number;
  predicted_hr?: number;
  predicted_rbi?: number;
  predicted_walks?: number;
  confidence?: number;
  has_prediction: boolean;
}

export interface GamePredictionsResult {
  game: GameDetail;
  home_players: GamePlayerPrediction[];
  away_players: GamePlayerPrediction[];
}

// Win Probability
export const getWinProbability = (gameId: number | string) =>
  fetchAPI<WinProbabilityResult>(`/v1/games/${gameId}/win-probability`);

// Accuracy
export const getAccuracySummary = () =>
  fetchAPI<AccuracySummary>("/v1/accuracy/summary");

export const getAccuracyByPlayer = (playerId: number) =>
  fetchAPI<PlayerAccuracy>(`/v1/accuracy/by-player/${playerId}`);

export const getCalibrationData = () =>
  fetchAPI<CalibrationPoint[]>("/v1/accuracy/calibration");

export const triggerBackfill = (lookbackDays = 7) =>
  fetchAPI<{ status: string }>(`/v1/accuracy/backfill?lookback_days=${lookbackDays}`, {
    method: "POST",
  });

// Win Probability Types
export interface TeamProjection {
  team_name: string;
  team_abbreviation: string;
  projected_runs: number;
  projected_hits: number;
  projected_hr: number;
  projected_rbi: number;
  projected_walks: number;
  n_players_with_predictions: number;
  n_total_players: number;
}

export interface WinProbabilityResult {
  home_win_pct: number;
  away_win_pct: number;
  home: TeamProjection;
  away: TeamProjection;
  confidence: number;
  method: string;
}

// Accuracy Types
export interface AccuracySummary {
  total_evaluated: number;
  avg_mse?: number;
  avg_mae?: number;
  hit_rate?: number;
  per_stat: Record<string, Record<string, number>>;
}

export interface PlayerAccuracyPrediction {
  prediction_id: number;
  date?: string;
  predicted: Record<string, number>;
  actual: Record<string, number>;
  mse?: number;
}

export interface PlayerAccuracy {
  player_id: number;
  total_evaluated: number;
  avg_mse?: number;
  predictions: PlayerAccuracyPrediction[];
}

export interface CalibrationPoint {
  confidence_bin: number;
  predicted_accuracy: number;
  actual_accuracy?: number;
  n_predictions: number;
}

// Leaderboard Types
export interface LeaderboardEntry {
  rank: number;
  player_id: number;
  player_name: string;
  team?: string;
  headshot_url?: string;
  predicted_hits: number;
  predicted_hr: number;
  predicted_rbi: number;
  predicted_walks: number;
  confidence?: number;
  composite_score: number;
}

// Pitcher Types
export interface PitcherStatsResult {
  id: number;
  mlb_id: number;
  name: string;
  team?: string;
  throws?: string;
  headshot_url?: string;
  stats: {
    total_games: number;
    total_innings: number;
    total_strikeouts: number;
    total_walks: number;
    total_earned_runs: number;
    era: number | null;
    whip: number | null;
    k_per_9: number | null;
    bb_per_9: number | null;
  };
}

// Player Compare
export interface PlayerCompareResult {
  players: PlayerDetail[];
}
