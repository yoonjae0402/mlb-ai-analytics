"use client";
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getWeightSensitivity, getModelMetrics } from "@/lib/api";
import WeightSensitivity from "@/components/charts/WeightSensitivity";
import MetricCard from "@/components/cards/MetricCard";
import { formatMetric } from "@/lib/utils";
import PageIntro from "@/components/ui/PageIntro";
import InfoTooltip from "@/components/ui/InfoTooltip";
import { Layers } from "lucide-react";

const MODEL_LABELS: Record<string, string> = {
  lstm: "LSTM",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  linear: "Linear",
};

const MODEL_COLORS: Record<string, string> = {
  lstm: "text-mlb-blue",
  xgboost: "text-mlb-red",
  lightgbm: "text-mlb-green",
  linear: "text-yellow-400",
};

const MODEL_BAR_COLORS: Record<string, string> = {
  lstm: "bg-mlb-blue",
  xgboost: "bg-mlb-red",
  lightgbm: "bg-mlb-green",
  linear: "bg-yellow-400",
};

export default function EnsemblePage() {
  const [strategy, setStrategy] = useState("weighted_average");
  const [rawWeights, setRawWeights] = useState<Record<string, number>>({
    lstm: 0.5,
    xgboost: 0.5,
    lightgbm: 0.5,
    linear: 0.5,
  });

  const { data: sensitivity, error: sensError } = useQuery({
    queryKey: ["weightSensitivity"],
    queryFn: getWeightSensitivity,
  });

  const { data: metrics } = useQuery({
    queryKey: ["modelMetrics"],
    queryFn: getModelMetrics,
  });

  const trainedModels: string[] = useMemo(() => {
    if (!metrics) return [];
    return ["lstm", "xgboost", "lightgbm", "linear"].filter(
      (k) => (metrics as Record<string, unknown>)?.[k]
    );
  }, [metrics]);

  const normalizedWeights = useMemo(() => {
    if (trainedModels.length === 0) return {};
    const total = trainedModels.reduce((s, k) => s + (rawWeights[k] ?? 0), 0);
    if (total === 0)
      return Object.fromEntries(trainedModels.map((k) => [k, 1 / trainedModels.length]));
    return Object.fromEntries(
      trainedModels.map((k) => [k, (rawWeights[k] ?? 0) / total])
    );
  }, [rawWeights, trainedModels]);

  // LSTM vs XGBoost weight for sensitivity chart
  const lstmRaw = rawWeights.lstm ?? 0.5;
  const xgbRaw = rawWeights.xgboost ?? 0.5;
  const lstmWeight = lstmRaw / Math.max(lstmRaw + xgbRaw, 0.001);

  const currentMSE = sensitivity?.sweep?.find(
    (s: { lstm_weight: number; mse: number }) => Math.abs(s.lstm_weight - lstmWeight) < 0.06
  )?.mse;

  const bestPoint = sensitivity?.sweep?.reduce(
    (
      best: { lstm_weight: number; mse: number },
      s: { lstm_weight: number; mse: number }
    ) => (s.mse < best.mse ? s : best),
    { lstm_weight: 0.5, mse: Infinity }
  );

  if (sensError && trainedModels.length === 0) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-8 text-center">
          <p className="text-mlb-muted">
            Train at least two models first to use the Ensemble Lab.
          </p>
        </div>
      </div>
    );
  }

  const bestSingleMSE = Math.min(
    ...trainedModels.map(
      (k) => (metrics as Record<string, Record<string, number>>)?.[k]?.mse ?? Infinity
    )
  );

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <PageIntro
        title="Combine Models for Better Predictions"
        icon={<Layers className="w-5 h-5" />}
        pageKey="ensemble"
      >
        <p>
          Blending predictions from multiple models often beats any single model alone.
          Use <strong>Weighted Average</strong> to control each model&apos;s contribution,
          or <strong>Stacking</strong> to let a meta-learner find the best mix automatically.
        </p>
      </PageIntro>

      {/* Strategy Controls */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-4" data-tour="ensemble-controls">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="text-xs text-mlb-muted block mb-2">
              Strategy<InfoTooltip term="ensemble" />
            </label>
            <div className="flex gap-2">
              {["weighted_average", "stacking"].map((s) => (
                <button
                  key={s}
                  onClick={() => setStrategy(s)}
                  className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                    strategy === s
                      ? "bg-mlb-blue text-white"
                      : "bg-mlb-surface text-mlb-muted hover:text-mlb-text"
                  }`}
                >
                  {s === "weighted_average" ? "Weighted Average" : "Stacking"}
                </button>
              ))}
            </div>
          </div>

          {strategy === "weighted_average" && trainedModels.length > 0 && (
            <div className="space-y-3">
              <p className="text-xs text-mlb-muted">Model Weights (drag to adjust)</p>
              {trainedModels.map((k) => (
                <div key={k}>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className={`font-semibold ${MODEL_COLORS[k]}`}>
                      {MODEL_LABELS[k]}
                    </span>
                    <span className="text-mlb-muted font-mono">
                      {((normalizedWeights[k] ?? 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={rawWeights[k] ?? 0.5}
                    onChange={(e) =>
                      setRawWeights((prev) => ({
                        ...prev,
                        [k]: Number(e.target.value),
                      }))
                    }
                    className="w-full"
                  />
                </div>
              ))}
              <p className="text-[10px] text-mlb-muted">
                Weights auto-normalize to sum to 100%.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Normalized weight bar */}
      {strategy === "weighted_average" && trainedModels.length > 1 && (
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
          <p className="text-xs text-mlb-muted mb-3 uppercase tracking-wider font-semibold">
            Active Blend
          </p>
          <div className="flex h-4 rounded-full overflow-hidden">
            {trainedModels.map((k) => {
              const pct = (normalizedWeights[k] ?? 0) * 100;
              return pct > 0.5 ? (
                <div
                  key={k}
                  className={`${MODEL_BAR_COLORS[k]} h-full transition-all duration-200`}
                  style={{ width: `${pct}%` }}
                  title={`${MODEL_LABELS[k]}: ${pct.toFixed(0)}%`}
                />
              ) : null;
            })}
          </div>
          <div className="flex flex-wrap gap-4 mt-2">
            {trainedModels.map((k) => (
              <div key={k} className="flex items-center gap-1">
                <div className={`w-2 h-2 rounded-full ${MODEL_BAR_COLORS[k]}`} />
                <span className="text-xs text-mlb-muted">
                  {MODEL_LABELS[k]}: {((normalizedWeights[k] ?? 0) * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metrics comparison */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {trainedModels.map((k) => {
          const m = (metrics as Record<string, Record<string, number>>)?.[k];
          return (
            <MetricCard
              key={k}
              label={
                <>
                  <span className={MODEL_COLORS[k]}>{MODEL_LABELS[k]}</span> MSE
                  <InfoTooltip term="mse" />
                </>
              }
              value={formatMetric(m?.mse)}
            />
          );
        })}
        {currentMSE !== undefined && (
          <MetricCard
            label={<>Ensemble MSE<InfoTooltip term="mse" /></>}
            value={formatMetric(currentMSE)}
            deltaType={currentMSE < bestSingleMSE ? "positive" : "neutral"}
            delta={currentMSE < bestSingleMSE ? "Better than all!" : undefined}
          />
        )}
        {bestPoint && bestPoint.mse < Infinity && (
          <MetricCard
            label="Best LSTM Weight"
            value={bestPoint.lstm_weight.toFixed(2)}
            delta={`MSE: ${formatMetric(bestPoint.mse)}`}
          />
        )}
      </div>

      {/* Weight Sensitivity Chart (LSTM vs XGBoost) */}
      {sensitivity &&
        trainedModels.includes("lstm") &&
        trainedModels.includes("xgboost") && (
          <div>
            <p className="text-xs text-mlb-muted mb-2">
              LSTM vs XGBoost sensitivity sweep — lower MSE is better
            </p>
            <WeightSensitivity sweep={sensitivity.sweep} />
          </div>
        )}
    </div>
  );
}
