/**
 * Stat helper utilities for beginner-friendly context.
 * Provides thresholds, percentile ratings, trend detection, and plain-English tooltips.
 */

export type ContextLevel = "elite" | "great" | "average" | "below_avg";

interface StatThresholds {
  elite: number;
  great: number;
  average: number;
  // Below average is anything under the average threshold
}

// Thresholds for common batting stats
const STAT_THRESHOLDS: Record<string, StatThresholds> = {
  batting_avg: { elite: 0.3, great: 0.27, average: 0.24 },
  obp: { elite: 0.38, great: 0.34, average: 0.31 },
  slg: { elite: 0.5, great: 0.44, average: 0.38 },
  woba: { elite: 0.37, great: 0.33, average: 0.3 },
  barrel_rate: { elite: 12, great: 8, average: 5 },
  exit_velocity: { elite: 92, great: 89, average: 87 },
  hard_hit_rate: { elite: 45, great: 38, average: 30 },
  k_rate: { elite: 15, great: 20, average: 25 }, // Lower is better
  bb_rate: { elite: 12, great: 9, average: 7 },
  sprint_speed: { elite: 29, great: 27.5, average: 26 },
  predicted_hits: { elite: 1.8, great: 1.3, average: 0.9 },
  predicted_hr: { elite: 0.3, great: 0.15, average: 0.08 },
  predicted_rbi: { elite: 1.5, great: 1.0, average: 0.6 },
  predicted_walks: { elite: 0.8, great: 0.5, average: 0.3 },
  home_runs: { elite: 35, great: 25, average: 15 },
  rbi: { elite: 100, great: 75, average: 50 },
  hits: { elite: 175, great: 150, average: 120 },
  walks: { elite: 80, great: 60, average: 40 },
};

// Stats where lower is better
const LOWER_IS_BETTER = new Set(["k_rate"]);

export function getContextLevel(stat: string, value: number): ContextLevel {
  const thresholds = STAT_THRESHOLDS[stat];
  if (!thresholds) return "average";

  if (LOWER_IS_BETTER.has(stat)) {
    if (value <= thresholds.elite) return "elite";
    if (value <= thresholds.great) return "great";
    if (value <= thresholds.average) return "average";
    return "below_avg";
  }

  if (value >= thresholds.elite) return "elite";
  if (value >= thresholds.great) return "great";
  if (value >= thresholds.average) return "average";
  return "below_avg";
}

export const CONTEXT_COLORS: Record<ContextLevel, { bg: string; text: string; label: string }> = {
  elite: { bg: "bg-mlb-gold/15", text: "text-mlb-gold", label: "Elite" },
  great: { bg: "bg-mlb-green/15", text: "text-mlb-green", label: "Great" },
  average: { bg: "bg-slate-500/15", text: "text-slate-400", label: "Average" },
  below_avg: { bg: "bg-mlb-orange/15", text: "text-mlb-orange", label: "Below Avg" },
};

// Percentile estimation (0-99 scale) based on MLB averages
const PERCENTILE_RANGES: Record<string, { min: number; max: number }> = {
  batting_avg: { min: 0.18, max: 0.34 },
  obp: { min: 0.25, max: 0.42 },
  slg: { min: 0.3, max: 0.6 },
  woba: { min: 0.25, max: 0.42 },
  barrel_rate: { min: 0, max: 18 },
  exit_velocity: { min: 83, max: 95 },
  hard_hit_rate: { min: 20, max: 55 },
  k_rate: { min: 30, max: 10 }, // Inverted: lower is better
  bb_rate: { min: 3, max: 16 },
  sprint_speed: { min: 24, max: 31 },
  predicted_hits: { min: 0, max: 2.5 },
  predicted_hr: { min: 0, max: 0.4 },
  predicted_rbi: { min: 0, max: 2 },
  predicted_walks: { min: 0, max: 1 },
};

export function getPercentile(stat: string, value: number): number {
  const range = PERCENTILE_RANGES[stat];
  if (!range) return 50;

  const { min, max } = range;
  const pct = ((value - min) / (max - min)) * 99;
  return Math.max(0, Math.min(99, Math.round(pct)));
}

export function getPercentileColor(percentile: number): string {
  if (percentile >= 90) return "bg-mlb-gold";
  if (percentile >= 70) return "bg-mlb-green";
  if (percentile >= 50) return "bg-slate-400";
  if (percentile >= 30) return "bg-mlb-orange";
  return "bg-mlb-red";
}

export function getPercentileTextColor(percentile: number): string {
  if (percentile >= 90) return "text-mlb-gold";
  if (percentile >= 70) return "text-mlb-green";
  if (percentile >= 50) return "text-slate-400";
  if (percentile >= 30) return "text-mlb-orange";
  return "text-mlb-red";
}

export type TrendDirection = "up" | "down" | "flat";

export function getTrend(recent: number, season: number): TrendDirection {
  if (season === 0) return "flat";
  const diff = (recent - season) / season;
  if (diff > 0.05) return "up";
  if (diff < -0.05) return "down";
  return "flat";
}

// Plain-English stat explanations
export const STAT_TOOLTIPS: Record<string, string> = {
  batting_avg: "How often a batter gets a hit. .300+ is excellent, .250 is average.",
  obp: "How often a batter reaches base (hits + walks). Higher means more opportunities to score.",
  slg: "Measures power — extra-base hits count more. A .500+ SLG means serious power.",
  woba: "The best single number for overall hitting. Weights all outcomes by their run value. .320 is average.",
  barrel_rate: "% of batted balls hit at the ideal speed and angle for damage. 8%+ is great.",
  exit_velocity: "How hard the ball comes off the bat. 90+ mph is elite contact quality.",
  hard_hit_rate: "% of balls hit 95+ mph. More hard contact = more hits and extra bases.",
  k_rate: "Strikeout percentage. Lower is better — under 20% means good bat-to-ball skills.",
  bb_rate: "Walk percentage. Higher means better plate discipline and pitch recognition.",
  sprint_speed: "Running speed in feet/second. 28+ ft/s is fast enough to beat out grounders.",
  launch_angle: "The vertical angle the ball leaves the bat. 10-25 degrees is the sweet spot for power.",
  park_factor: "How much the ballpark affects scoring. Over 100 = hitter-friendly park.",
  predicted_hits: "AI-predicted hits for the next game based on recent performance and matchup.",
  predicted_hr: "AI-predicted home run probability. Even 0.15+ is a good sign.",
  predicted_rbi: "AI-predicted runs batted in for the next game.",
  predicted_walks: "AI-predicted walks based on plate discipline patterns.",
};

// Display-friendly stat names
export const STAT_DISPLAY: Record<string, string> = {
  batting_avg: "AVG",
  obp: "OBP",
  slg: "SLG",
  woba: "wOBA",
  barrel_rate: "Barrel %",
  exit_velocity: "Exit Velo",
  hard_hit_rate: "Hard Hit %",
  k_rate: "K%",
  bb_rate: "BB%",
  sprint_speed: "Sprint Spd",
  launch_angle: "Launch Angle",
  park_factor: "Park Factor",
  predicted_hits: "Pred. Hits",
  predicted_hr: "Pred. HR",
  predicted_rbi: "Pred. RBI",
  predicted_walks: "Pred. BB",
  home_runs: "HR",
  rbi: "RBI",
  hits: "Hits",
  walks: "BB",
};

export function formatStatValue(stat: string, value: number): string {
  if (["batting_avg", "obp", "slg", "woba"].includes(stat)) {
    return value.toFixed(3);
  }
  if (["predicted_hits", "predicted_hr", "predicted_rbi", "predicted_walks"].includes(stat)) {
    return value.toFixed(2);
  }
  if (["barrel_rate", "k_rate", "bb_rate", "hard_hit_rate"].includes(stat)) {
    return `${value.toFixed(1)}%`;
  }
  if (stat === "exit_velocity" || stat === "sprint_speed") {
    return value.toFixed(1);
  }
  return String(Math.round(value));
}
