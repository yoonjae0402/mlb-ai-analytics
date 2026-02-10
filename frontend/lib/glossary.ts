export interface GlossaryEntry {
  term: string;
  definition: string;
}

const glossary: Record<string, GlossaryEntry> = {
  // Baseball stats
  batting_avg: {
    term: "Batting Average (AVG)",
    definition:
      "Hits divided by at-bats. A .300 AVG is excellent; .250 is roughly league average.",
  },
  obp: {
    term: "On-Base Percentage (OBP)",
    definition:
      "How often a batter reaches base. Includes hits, walks, and hit-by-pitches. League average is around .320.",
  },
  slg: {
    term: "Slugging Percentage (SLG)",
    definition:
      "Total bases divided by at-bats. Measures power — a higher SLG means more extra-base hits.",
  },
  woba: {
    term: "Weighted On-Base Average (wOBA)",
    definition:
      "An advanced stat that weights each way of reaching base by its actual run value. Better than AVG or OBP alone. League average is about .320.",
  },
  barrel_rate: {
    term: "Barrel Rate",
    definition:
      "Percentage of batted balls hit at an ideal combination of exit velocity (95+ mph) and launch angle (26-30°). Barrels almost always result in hits or extra bases.",
  },
  exit_velocity: {
    term: "Exit Velocity",
    definition:
      "How fast the ball comes off the bat in mph. Higher = harder contact. League average is ~88 mph; elite is 92+.",
  },
  launch_angle: {
    term: "Launch Angle",
    definition:
      "The vertical angle the ball leaves the bat. 10-25° is ideal for line drives and fly balls. Too high = pop-ups; too low = grounders.",
  },
  sprint_speed: {
    term: "Sprint Speed",
    definition:
      "A player's top running speed in feet per second, measured by Statcast. 30+ ft/s is elite.",
  },
  k_rate: {
    term: "Strikeout Rate (K%)",
    definition:
      "Percentage of plate appearances that end in a strikeout. Lower is better for hitters — league average is around 22%.",
  },
  bb_rate: {
    term: "Walk Rate (BB%)",
    definition:
      "Percentage of plate appearances that result in a walk. Higher shows better plate discipline. 10%+ is good.",
  },
  hard_hit_rate: {
    term: "Hard Hit Rate",
    definition:
      "Percentage of batted balls with an exit velocity of 95+ mph. A good proxy for consistent hard contact.",
  },
  park_factor: {
    term: "Park Factor",
    definition:
      "Adjusts stats for the ballpark. A factor above 1.0 means the park favors hitters; below 1.0 favors pitchers.",
  },
  platoon_advantage: {
    term: "Platoon Advantage",
    definition:
      "The statistical edge a batter has when facing a pitcher who throws from the opposite side (e.g., lefty batter vs. righty pitcher).",
  },
  home_runs: {
    term: "Home Runs (HR)",
    definition:
      "Balls hit out of the field of play, scoring the batter and all runners. 30+ HR in a season is a power hitter.",
  },
  rbi: {
    term: "Runs Batted In (RBI)",
    definition:
      "The number of runs a batter drives in with hits, walks, or sacrifice plays. Depends on team context.",
  },
  hits: {
    term: "Hits",
    definition:
      "Any time the batter reaches base safely on a batted ball (single, double, triple, or home run).",
  },
  walks: {
    term: "Walks (BB)",
    definition:
      "When a batter receives 4 balls and is awarded first base. Shows patience and plate discipline.",
  },
  win_probability: {
    term: "Win Probability",
    definition:
      "The estimated chance a team will win, based on the current score, inning, base-runners, and outs.",
  },

  // ML / Model terms
  lstm: {
    term: "LSTM (Long Short-Term Memory)",
    definition:
      "A type of neural network designed for sequences. It reads a player's last 10 games in order and learns patterns over time — like a hot streak or slump.",
  },
  xgboost: {
    term: "XGBoost",
    definition:
      "A machine-learning model that uses many decision trees working together. It looks at summary statistics (averages, trends) rather than game-by-game sequences.",
  },
  ensemble: {
    term: "Ensemble",
    definition:
      "Combining two or more models to get a better prediction. Like asking multiple experts and averaging their opinions.",
  },
  attention: {
    term: "Attention Mechanism",
    definition:
      "A technique that lets the model focus on the most relevant past games instead of weighting them all equally. Brighter attention = the model thinks that game matters more.",
  },
  mse: {
    term: "Mean Squared Error (MSE)",
    definition:
      "Measures prediction accuracy — the average of (predicted - actual) squared. Lower is better. Penalizes big misses more than small ones.",
  },
  mae: {
    term: "Mean Absolute Error (MAE)",
    definition:
      "The average absolute difference between predictions and actual values. Easier to interpret than MSE — an MAE of 0.5 means predictions are off by 0.5 on average.",
  },
  r2: {
    term: "R-squared (R²)",
    definition:
      "How much of the variation in actual values the model explains. 1.0 = perfect predictions; 0 = no better than guessing the average; negative = worse than guessing.",
  },
  epochs: {
    term: "Epochs",
    definition:
      "The number of times the model trains on the entire dataset. More epochs can improve accuracy, but too many risk overfitting.",
  },
  learning_rate: {
    term: "Learning Rate",
    definition:
      "How big of a step the model takes when updating its weights. Too high = unstable; too low = slow to learn.",
  },
  hidden_size: {
    term: "Hidden Size",
    definition:
      "The number of neurons in the LSTM's memory. Larger = more capacity to learn patterns, but slower and needs more data.",
  },
  batch_size: {
    term: "Batch Size",
    definition:
      "How many training examples the model processes at once before updating. Larger batches are more stable but use more memory.",
  },
  overfitting: {
    term: "Overfitting",
    definition:
      "When a model memorizes training data instead of learning general patterns. It performs well on training data but poorly on new data.",
  },
  weighted_average: {
    term: "Weighted Average (Ensemble)",
    definition:
      "Combines two model predictions using a weight slider. For example, 0.6 LSTM + 0.4 XGBoost gives LSTM slightly more influence.",
  },
  stacking: {
    term: "Stacking (Ensemble)",
    definition:
      "A meta-learner (simple model) that learns the best way to combine LSTM and XGBoost predictions from data, rather than using fixed weights.",
  },
};

export default glossary;
