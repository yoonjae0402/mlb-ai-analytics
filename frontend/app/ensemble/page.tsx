"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getWeightSensitivity, getModelMetrics } from "@/lib/api";
import WeightSensitivity from "@/components/charts/WeightSensitivity";
import MetricCard from "@/components/cards/MetricCard";
import { formatMetric } from "@/lib/utils";
import PageIntro from "@/components/ui/PageIntro";
import InfoTooltip from "@/components/ui/InfoTooltip";
import { Layers } from "lucide-react";

export default function EnsemblePage() {
  const [strategy, setStrategy] = useState("weighted_average");
  const [lstmWeight, setLstmWeight] = useState(0.5);

  const { data: sensitivity, error: sensError } = useQuery({
    queryKey: ["weightSensitivity"],
    queryFn: getWeightSensitivity,
  });

  const { data: metrics } = useQuery({
    queryKey: ["modelMetrics"],
    queryFn: getModelMetrics,
  });

  if (sensError) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-8 text-center">
          <p className="text-mlb-muted">
            Train both models first to use the Ensemble Lab.
          </p>
        </div>
      </div>
    );
  }

  // Find the ensemble MSE at current weight
  const currentMSE = sensitivity?.sweep?.find(
    (s) => Math.abs(s.lstm_weight - lstmWeight) < 0.06
  )?.mse;
  const bestPoint = sensitivity?.sweep?.reduce(
    (best, s) => (s.mse < best.mse ? s : best),
    { lstm_weight: 0.5, mse: Infinity }
  );

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <PageIntro title="Combine Models for Better Predictions" icon={<Layers className="w-5 h-5" />} pageKey="ensemble">
        <p>
          Blending predictions from LSTM and XGBoost often beats either model alone — like
          asking two experts and averaging their answers. Use <strong>Weighted Average</strong> to
          control the blend, or <strong>Stacking</strong> to let a meta-learner figure out the best mix.
        </p>
      </PageIntro>

      {/* Strategy Controls */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-4" data-tour="ensemble-controls">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="text-xs text-mlb-muted block mb-2">Strategy<InfoTooltip term="ensemble" /></label>
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

          {strategy === "weighted_average" && (
            <div>
              <label className="text-xs text-mlb-muted block mb-2">
                LSTM Weight: {lstmWeight.toFixed(2)} / XGBoost: {(1 - lstmWeight).toFixed(2)}
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={lstmWeight}
                onChange={(e) => setLstmWeight(Number(e.target.value))}
                className="w-full"
              />
            </div>
          )}
        </div>
      </div>

      {/* Metrics comparison */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {metrics?.lstm && (
          <MetricCard
            label={<>LSTM MSE<InfoTooltip term="mse" /></>}
            value={formatMetric(metrics.lstm.mse)}
          />
        )}
        {metrics?.xgboost && (
          <MetricCard
            label={<>XGBoost MSE<InfoTooltip term="mse" /></>}
            value={formatMetric(metrics.xgboost.mse)}
          />
        )}
        {currentMSE !== undefined && (
          <MetricCard
            label={<>Ensemble MSE<InfoTooltip term="mse" /></>}
            value={formatMetric(currentMSE)}
            deltaType={
              currentMSE <
              Math.min(metrics?.lstm?.mse ?? Infinity, metrics?.xgboost?.mse ?? Infinity)
                ? "positive"
                : "neutral"
            }
            delta={
              currentMSE <
              Math.min(metrics?.lstm?.mse ?? Infinity, metrics?.xgboost?.mse ?? Infinity)
                ? "Better than both!"
                : undefined
            }
          />
        )}
        {bestPoint && bestPoint.mse < Infinity && (
          <MetricCard
            label="Optimal Weight"
            value={bestPoint.lstm_weight.toFixed(2)}
            delta={`MSE: ${formatMetric(bestPoint.mse)}`}
          />
        )}
      </div>

      {/* Weight Sensitivity Chart */}
      {sensitivity && <WeightSensitivity sweep={sensitivity.sweep} />}
    </div>
  );
}
