"use client";
import { Database, Server, Monitor, Brain, Zap, FileCode } from "lucide-react";
import PageIntro from "@/components/ui/PageIntro";

const techStack = [
  { category: "Frontend", items: "Next.js 14, TypeScript, Tailwind CSS, Recharts, React Query" },
  { category: "Backend", items: "FastAPI, SQLAlchemy, Pydantic, Uvicorn" },
  { category: "Database", items: "PostgreSQL, Alembic migrations" },
  { category: "ML/AI", items: "PyTorch (BiLSTM + Attention), XGBoost, Optuna, scikit-learn" },
  { category: "Data Sources", items: "pybaseball (Statcast, FanGraphs), MLB Stats API" },
  { category: "Deployment", items: "Docker, Vercel (frontend), Railway (backend)" },
];

const pipeline = [
  { icon: Database, label: "Data Sources", desc: "pybaseball + MLB Stats API", color: "text-mlb-blue" },
  { icon: Zap, label: "Feature Engineering", desc: "22 features: Statcast + pitcher matchup + derived", color: "text-yellow-400" },
  { icon: Brain, label: "Models", desc: "BiLSTM + Attention, XGBoost, Ensemble", color: "text-mlb-red" },
  { icon: Server, label: "FastAPI", desc: "REST API with async endpoints", color: "text-mlb-green" },
  { icon: Monitor, label: "Next.js", desc: "Interactive dashboard + visualizations", color: "text-purple-400" },
];

const models = [
  {
    name: "PlayerLSTM",
    desc: "Bidirectional LSTM with multi-head self-attention for sequential player performance prediction.",
    details: [
      "2 LSTM layers, 128 hidden units (bidirectional)",
      "8-head self-attention mechanism",
      "LayerNorm + GELU activation + Dropout",
      "Xavier/Glorot weight initialization",
      "Input: (batch, 10, 26) → Output: (batch, 4)",
      "26 features: 15 batter + 5 pitcher matchup + 6 derived/context",
      "Monte Carlo Dropout for uncertainty estimation (90% CI)",
    ],
  },
  {
    name: "XGBoostPredictor",
    desc: "Gradient-boosted trees with sequence flattening via summary statistics.",
    details: [
      "Per-feature: mean, std, last value, linear trend slope",
      "Flattens (n, 10, 26) → (n, 104)",
      "Separate XGBRegressor per target",
      "Default: 200 estimators, max_depth=6",
    ],
  },
  {
    name: "EnsemblePredictor",
    desc: "Combines LSTM + XGBoost predictions for improved accuracy.",
    details: [
      "Weighted average with tunable weights",
      "Ridge regression stacking meta-learner",
      "Weight sensitivity analysis",
    ],
  },
];

export default function ArchitecturePage() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <PageIntro title="How the Platform is Built" icon={<FileCode className="w-5 h-5" />} pageKey="architecture">
        <p>
          Technical documentation for students and developers. See the full data pipeline,
          tech stack, model architectures, and API endpoints that power the platform.
        </p>
      </PageIntro>

      {/* Architecture Diagram */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-6">
        <h3 className="text-sm font-semibold text-mlb-text mb-6">
          System Architecture
        </h3>
        <div className="flex items-center justify-between gap-2 overflow-x-auto">
          {pipeline.map((step, i) => {
            const Icon = step.icon;
            return (
              <div key={step.label} className="flex items-center gap-2">
                <div className="text-center min-w-[120px]">
                  <div className="w-12 h-12 rounded-xl bg-mlb-surface flex items-center justify-center mx-auto mb-2">
                    <Icon className={`w-6 h-6 ${step.color}`} />
                  </div>
                  <p className="text-xs font-semibold text-mlb-text">
                    {step.label}
                  </p>
                  <p className="text-[10px] text-mlb-muted mt-0.5">
                    {step.desc}
                  </p>
                </div>
                {i < pipeline.length - 1 && (
                  <div className="text-mlb-muted text-lg">→</div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Tech Stack */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-6">
        <h3 className="text-sm font-semibold text-mlb-text mb-4">
          Tech Stack
        </h3>
        <div className="space-y-3">
          {techStack.map((row) => (
            <div key={row.category} className="flex gap-4">
              <span className="text-xs font-semibold text-mlb-blue w-24 flex-shrink-0">
                {row.category}
              </span>
              <span className="text-xs text-mlb-muted">{row.items}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Model Documentation */}
      <div className="space-y-4">
        <h3 className="text-sm font-semibold text-mlb-text">
          Model Documentation
        </h3>
        {models.map((model) => (
          <details
            key={model.name}
            className="bg-mlb-card border border-mlb-border rounded-xl group"
          >
            <summary className="p-4 cursor-pointer text-sm font-semibold text-mlb-text hover:text-mlb-blue transition-colors">
              {model.name}
            </summary>
            <div className="px-4 pb-4 border-t border-mlb-border pt-3">
              <p className="text-xs text-mlb-muted mb-3">{model.desc}</p>
              <ul className="space-y-1">
                {model.details.map((d, i) => (
                  <li key={i} className="text-xs text-mlb-muted flex gap-2">
                    <span className="text-mlb-blue">-</span>
                    {d}
                  </li>
                ))}
              </ul>
            </div>
          </details>
        ))}
      </div>

      {/* API Reference */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-6">
        <h3 className="text-sm font-semibold text-mlb-text mb-4">
          API Reference
        </h3>
        <div className="space-y-2 font-mono text-xs">
          {[
            ["GET", "/health", "Health check"],
            ["POST", "/v1/train", "Train LSTM + XGBoost"],
            ["GET", "/v1/train/status", "Training progress"],
            ["GET", "/v1/train/curves", "Loss curves"],
            ["POST", "/v1/predict/player", "Predict next game"],
            ["GET", "/v1/players/search?q=", "Search players"],
            ["GET", "/v1/players/{id}", "Player profile"],
            ["POST", "/v1/attention/weights", "Attention heatmap"],
            ["POST", "/v1/attention/feature-attribution", "Feature importance"],
            ["POST", "/v1/ensemble/predict", "Ensemble prediction"],
            ["GET", "/v1/ensemble/weight-sensitivity", "Weight sweep"],
            ["GET", "/v1/games/live", "Live games"],
            ["GET", "/v1/model/metrics", "Model metrics"],
            ["GET", "/v1/model/evaluation", "Full evaluation"],
            ["GET", "/v1/data/status", "Data freshness"],
            ["POST", "/v1/data/refresh", "Refresh data"],
            ["POST", "/v1/tune", "Start Optuna tuning"],
            ["GET", "/v1/games/{id}/win-probability", "Win probability"],
            ["GET", "/v1/schedule/range", "Schedule calendar"],
            ["GET", "/v1/accuracy/summary", "Prediction accuracy"],
            ["GET", "/v1/leaderboard", "Player leaderboard"],
            ["GET", "/v1/pitchers/search", "Search pitchers"],
            ["GET", "/v1/pitchers/{id}/stats", "Pitcher stats"],
          ].map(([method, path, desc]) => (
            <div key={path} className="flex items-center gap-3">
              <span
                className={`w-12 text-center text-[10px] font-bold px-1 py-0.5 rounded ${
                  method === "GET"
                    ? "bg-mlb-green/20 text-mlb-green"
                    : "bg-mlb-blue/20 text-mlb-blue"
                }`}
              >
                {method}
              </span>
              <span className="text-mlb-text flex-1">{path}</span>
              <span className="text-mlb-muted">{desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
