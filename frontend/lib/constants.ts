export const TEAM_COLORS: Record<string, { primary: string; secondary: string }> = {
  NYY: { primary: "#003087", secondary: "#E4002C" },
  BOS: { primary: "#BD3039", secondary: "#0C2340" },
  LAD: { primary: "#005A9C", secondary: "#EF3E42" },
  SF: { primary: "#FD5A1E", secondary: "#27251F" },
  HOU: { primary: "#002D62", secondary: "#EB6E1F" },
  ATL: { primary: "#CE1141", secondary: "#13274F" },
  NYM: { primary: "#002D72", secondary: "#FF5910" },
  PHI: { primary: "#E81828", secondary: "#002D72" },
  SD: { primary: "#2F241D", secondary: "#FFC425" },
  CHC: { primary: "#0E3386", secondary: "#CC3433" },
  SEA: { primary: "#0C2C56", secondary: "#005C5C" },
  TB: { primary: "#092C5C", secondary: "#8FBCE6" },
  MIN: { primary: "#002B5C", secondary: "#D31145" },
  TEX: { primary: "#003278", secondary: "#C0111F" },
  BAL: { primary: "#DF4601", secondary: "#27251F" },
  CLE: { primary: "#00385D", secondary: "#E31937" },
  DET: { primary: "#0C2340", secondary: "#FA4616" },
  KC: { primary: "#004687", secondary: "#BD9B60" },
  LAA: { primary: "#BA0021", secondary: "#003263" },
  OAK: { primary: "#003831", secondary: "#EFB21E" },
  CWS: { primary: "#27251F", secondary: "#C4CED4" },
  PIT: { primary: "#27251F", secondary: "#FDB827" },
  STL: { primary: "#C41E3A", secondary: "#0C2340" },
  MIL: { primary: "#12284B", secondary: "#FFC52F" },
  CIN: { primary: "#C6011F", secondary: "#000000" },
  CHW: { primary: "#27251F", secondary: "#C4CED4" },
  COL: { primary: "#33006F", secondary: "#C4CED4" },
  ARI: { primary: "#A71930", secondary: "#E3D4AD" },
  MIA: { primary: "#00A3E0", secondary: "#EF3340" },
  WAS: { primary: "#AB0003", secondary: "#14225A" },
  TOR: { primary: "#134A8E", secondary: "#E8291C" },
};

export const FEATURE_DISPLAY_NAMES: Record<string, string> = {
  batting_avg: "Batting Avg",
  on_base_pct: "OBP",
  slugging_pct: "SLG",
  woba: "wOBA",
  barrel_rate: "Barrel %",
  exit_velocity: "Exit Velo",
  launch_angle: "Launch Angle",
  sprint_speed: "Sprint Speed",
  k_rate: "K%",
  bb_rate: "BB%",
  hard_hit_rate: "Hard Hit %",
  pull_rate: "Pull %",
  park_factor: "Park Factor",
  platoon_advantage: "Platoon Adv",
  days_rest: "Days Rest",
  opp_era: "Opp ERA",
  opp_whip: "Opp WHIP",
  opp_k_per_9: "Opp K/9",
  opp_bb_per_9: "Opp BB/9",
  opp_handedness_adv: "Matchup Adv",
  iso: "ISO",
  hot_streak: "Hot Streak",
  babip: "BABIP",
  cold_streak: "Cold Streak",
  is_home: "Home/Away",
  opp_quality: "Opp Win %",
};

export const TARGET_DISPLAY_NAMES: Record<string, string> = {
  hits: "Hits",
  home_runs: "Home Runs",
  rbi: "RBI",
  walks: "Walks",
};

export const NAV_SECTIONS = [
  {
    label: "Games",
    items: [
      { name: "Scores & Schedule", path: "/dashboard", icon: "Activity" },
      { name: "Schedule Calendar", path: "/dashboard/schedule", icon: "Calendar" },
    ],
  },
  {
    label: "Predictions",
    items: [
      { name: "Prediction Hub", path: "/dashboard/predictions", icon: "TrendingUp" },
      { name: "Leaderboard", path: "/dashboard/leaderboard", icon: "Trophy" },
      { name: "Player Predict", path: "/predict", icon: "Search" },
      { name: "Accuracy", path: "/dashboard/accuracy", icon: "Target" },
    ],
  },
  {
    label: "Players",
    items: [
      { name: "Player Index", path: "/dashboard/players", icon: "Users" },
      { name: "Pitcher Stats", path: "/dashboard/pitchers", icon: "TrendingUp" },
      { name: "Compare Players", path: "/dashboard/compare", icon: "Scale" },
    ],
  },
  {
    label: "Analysis",
    items: [
      { name: "Model Comparison", path: "/models", icon: "BarChart3" },
      { name: "Attention Viz", path: "/attention", icon: "Eye" },
      { name: "Ensemble Lab", path: "/ensemble", icon: "Layers" },
      { name: "Hyperparameter Tuning", path: "/tuning", icon: "Sliders" },
    ],
  },
  {
    label: "System",
    items: [
      { name: "Architecture", path: "/architecture", icon: "FileCode" },
      { name: "System Health", path: "/dashboard/system", icon: "Activity" },
    ],
  },
];

// Flat list for backwards compat (Header, etc.)
export const NAV_ITEMS = NAV_SECTIONS.flatMap((s) => s.items);
