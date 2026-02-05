"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getAttentionWeights, getFeatureAttribution } from "@/lib/api";
import AttentionHeatmap from "@/components/charts/AttentionHeatmap";
import FeatureImportance from "@/components/charts/FeatureImportance";
import MetricCard from "@/components/cards/MetricCard";
import { TARGET_DISPLAY_NAMES } from "@/lib/constants";

export default function AttentionPage() {
  const [sampleIdx, setSampleIdx] = useState(0);

  const { data: attention, isLoading: loadingAttn, error: attnError } = useQuery({
    queryKey: ["attention", sampleIdx],
    queryFn: () => getAttentionWeights(sampleIdx),
  });

  const { data: attribution } = useQuery({
    queryKey: ["attribution", sampleIdx],
    queryFn: () => getFeatureAttribution(sampleIdx),
  });

  if (attnError) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-8 text-center">
          <p className="text-mlb-muted">
            Train models first on the Model Comparison page to visualize attention weights.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Sample selector */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
        <div className="flex items-center gap-4">
          <label className="text-sm text-mlb-muted">Validation Sample:</label>
          <input
            type="range"
            min={0}
            max={(attention?.n_samples || 1) - 1}
            value={sampleIdx}
            onChange={(e) => setSampleIdx(Number(e.target.value))}
            className="flex-1"
          />
          <span className="text-sm font-mono text-mlb-text w-16 text-right">
            {sampleIdx + 1} / {attention?.n_samples || "—"}
          </span>
        </div>
      </div>

      {/* Predictions vs Actuals */}
      {attention && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {attention.target_names.map((name, i) => (
            <MetricCard
              key={name}
              label={TARGET_DISPLAY_NAMES[name] || name}
              value={attention.prediction[i]?.toFixed(2) || "—"}
              delta={`Actual: ${attention.actual[i]?.toFixed(2) || "—"}`}
              deltaType={
                Math.abs((attention.prediction[i] || 0) - (attention.actual[i] || 0)) < 0.5
                  ? "positive"
                  : "negative"
              }
            />
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Attention Heatmap */}
        {attention && attention.attention_weights[0] && (
          <AttentionHeatmap weights={attention.attention_weights[0]} />
        )}

        {/* Feature Importance */}
        {attribution && (
          <FeatureImportance
            importance={attribution.feature_importance}
            featureNames={attribution.feature_names}
          />
        )}
      </div>
    </div>
  );
}
