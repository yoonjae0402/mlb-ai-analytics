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
    { key: "predicted_hits", label: "Hits", value: result.predicted_hits, idx: 0 },
    { key: "predicted_hr", label: "Home Runs", value: result.predicted_hr, idx: 1 },
    { key: "predicted_rbi", label: "RBI", value: result.predicted_rbi, idx: 2 },
    { key: "predicted_walks", label: "Walks", value: result.predicted_walks, idx: 3 },
  ];

  const hasCi = result.confidence_interval_low && result.confidence_interval_high;

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
        {targets.map((t) => {
          const ciLow = hasCi ? result.confidence_interval_low![t.idx] : null;
          const ciHigh = hasCi ? result.confidence_interval_high![t.idx] : null;
          const unc = result.uncertainty?.[t.idx];
          // Uncertainty color: green = tight (<0.3), yellow = moderate, red = wide (>0.8)
          const uncColor = unc != null
            ? unc < 0.3 ? "text-mlb-green" : unc < 0.8 ? "text-yellow-400" : "text-mlb-red"
            : "";

          return (
            <div key={t.key} className="bg-mlb-card border border-mlb-border rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] text-mlb-muted uppercase tracking-wider">
                  {t.label}
                  <StatTooltip stat={t.key} />
                </span>
                <ContextBadge stat={t.key} value={t.value} />
              </div>
              <p className={`text-2xl font-bold mb-1 ${
                t.key === "predicted_hr" ? "text-mlb-red" : "text-mlb-text"
              }`}>
                {formatStatValue(t.key, t.value)}
              </p>

              {/* Confidence Interval */}
              {ciLow != null && ciHigh != null && (
                <div className="mb-2">
                  <div className="flex items-center justify-between text-[10px] mb-1">
                    <span className="text-mlb-muted">90% CI</span>
                    <span className={`font-mono font-semibold ${uncColor}`}>
                      {ciLow.toFixed(2)} – {ciHigh.toFixed(2)}
                    </span>
                  </div>
                  {/* CI visualization bar */}
                  <CIBar value={t.value} low={ciLow} high={ciHigh} />
                </div>
              )}

              <PercentileBar stat={t.key} value={t.value} />
            </div>
          );
        })}
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

function CIBar({ value, low, high }: { value: number; low: number; high: number }) {
  // Scale to a visual range — cap max at value * 3 or high * 1.5
  const maxScale = Math.max(high * 1.5, value * 3, 1);
  const leftPct = Math.max(0, (low / maxScale) * 100);
  const rightPct = Math.min(100, (high / maxScale) * 100);
  const valuePct = Math.min(100, (value / maxScale) * 100);
  const width = rightPct - leftPct;
  // Uncertainty color
  const range = high - low;
  const barColor = range < 0.5 ? "bg-mlb-green/30" : range < 1.5 ? "bg-yellow-400/30" : "bg-mlb-red/30";
  const dotColor = range < 0.5 ? "bg-mlb-green" : range < 1.5 ? "bg-yellow-400" : "bg-mlb-red";

  return (
    <div className="relative h-2 bg-mlb-surface rounded-full overflow-hidden">
      {/* CI range */}
      <div
        className={`absolute h-full rounded-full ${barColor}`}
        style={{ left: `${leftPct}%`, width: `${width}%` }}
      />
      {/* Point estimate */}
      <div
        className={`absolute top-0 w-1.5 h-full rounded-full ${dotColor}`}
        style={{ left: `${Math.max(0, valuePct - 0.75)}%` }}
      />
    </div>
  );
}
