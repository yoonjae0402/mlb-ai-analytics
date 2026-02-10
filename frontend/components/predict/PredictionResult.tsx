"use client";
import type { PredictionResult as PredResult } from "@/lib/api";
import ContextBadge from "@/components/ui/ContextBadge";
import PercentileBar from "@/components/ui/PercentileBar";
import StatTooltip from "@/components/ui/StatTooltip";
import { formatStatValue } from "@/lib/stat-helpers";

interface PredictionResultProps {
  result: PredResult;
}

export default function PredictionResultView({ result }: PredictionResultProps) {
  const targets = [
    { key: "predicted_hits", label: "Hits", value: result.predicted_hits },
    { key: "predicted_hr", label: "Home Runs", value: result.predicted_hr },
    { key: "predicted_rbi", label: "RBI", value: result.predicted_rbi },
    { key: "predicted_walks", label: "Walks", value: result.predicted_walks },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-mlb-text">
          Predicted Next-Game Stats
        </h3>
        <span className="text-xs text-mlb-muted px-2 py-1 bg-mlb-surface rounded-full">
          {result.model_type.toUpperCase()}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {targets.map((t) => (
          <div key={t.key} className="bg-mlb-card border border-mlb-border rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-mlb-muted uppercase tracking-wider">
                {t.label}
                <StatTooltip stat={t.key} />
              </span>
              <ContextBadge stat={t.key} value={t.value} />
            </div>
            <p className={`text-2xl font-bold mb-2 ${
              t.key === "predicted_hr" ? "text-mlb-red" : "text-mlb-text"
            }`}>
              {formatStatValue(t.key, t.value)}
            </p>
            <PercentileBar stat={t.key} value={t.value} />
          </div>
        ))}
      </div>

      {Object.keys(result.all_predictions).length > 1 && (
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-3">
          <p className="text-xs text-mlb-muted mb-2">All Model Predictions</p>
          <div className="space-y-1">
            {Object.entries(result.all_predictions).map(([model, preds]) => (
              <div key={model} className="flex justify-between text-xs">
                <span className="text-mlb-muted uppercase">{model}</span>
                <span className="text-mlb-text font-mono">
                  {(preds as number[]).map((p) => p.toFixed(2)).join(" / ")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
