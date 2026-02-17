"use client";
import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { startTraining, getTrainCurves, getModelMetrics, getModelEvaluation, type TrainConfig } from "@/lib/api";
import { useTrainingStatus } from "@/hooks/useTrainingStatus";
import TrainControls from "@/components/train/TrainControls";
import TrainProgress from "@/components/train/TrainProgress";
import TrainingCurvesChart from "@/components/charts/TrainingCurves";
import BarComparison from "@/components/charts/BarComparison";
import MetricCard from "@/components/cards/MetricCard";
import { formatMetric } from "@/lib/utils";
import PageIntro from "@/components/ui/PageIntro";
import InfoTooltip from "@/components/ui/InfoTooltip";
import { Brain } from "lucide-react";

const MODEL_COLORS: Record<string, string> = {
  lstm: "text-mlb-blue",
  xgboost: "text-mlb-red",
  lightgbm: "text-mlb-green",
  linear: "text-yellow-400",
};

const MODEL_LABELS: Record<string, string> = {
  lstm: "LSTM",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  linear: "Linear",
};

export default function ModelsPage() {
  const [trainLightGBM, setTrainLightGBM] = useState(false);
  const [trainLinear, setTrainLinear] = useState(false);

  const { data: status } = useTrainingStatus(true);
  const { data: curves } = useQuery({
    queryKey: ["trainCurves"],
    queryFn: getTrainCurves,
    refetchInterval: status?.is_training ? 2000 : false,
  });
  const { data: metrics } = useQuery({
    queryKey: ["modelMetrics"],
    queryFn: getModelMetrics,
    refetchInterval: status?.is_training ? 2000 : false,
  });

  const { data: evaluation } = useQuery({
    queryKey: ["modelEvaluation"],
    queryFn: getModelEvaluation,
    enabled: !!metrics,
    retry: false,
  });

  const trainMutation = useMutation({
    mutationFn: (config: TrainConfig) => startTraining(config),
  });

  const handleTrain = (config: TrainConfig) => {
    trainMutation.mutate({
      ...config,
      train_lightgbm: trainLightGBM,
      train_linear: trainLinear,
    });
  };

  // All model keys that have metrics
  const modelKeys = ["lstm", "xgboost", "lightgbm", "linear"].filter(
    (k) => (metrics as Record<string, unknown>)?.[k]
  );

  // Build per-target comparison data across all trained models
  const perTargetData =
    modelKeys.length > 0
      ? ["hits", "home_runs", "rbi", "walks"].map((target) => {
          const entry: Record<string, string | number> = {
            name: target.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()),
          };
          for (const k of modelKeys) {
            entry[MODEL_LABELS[k]] = (metrics as Record<string, Record<string, Record<string, Record<string, number>>>>)?.[k]?.per_target?.[target]?.mse ?? 0;
          }
          return entry;
        })
      : [];

  const comparisonRows = modelKeys.map((k) => {
    const m = (metrics as Record<string, Record<string, number>>)?.[k];
    return {
      key: k,
      label: MODEL_LABELS[k],
      mse: m?.mse,
      mae: m?.mae,
      r2: m?.r2,
    };
  });

  const bestMSE = comparisonRows.length
    ? Math.min(...comparisonRows.map((r) => r.mse ?? Infinity))
    : Infinity;

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <PageIntro title="Train & Compare AI Models" icon={<Brain className="w-5 h-5" />} pageKey="models">
        <p>
          Train multiple AI models on real MLB Statcast data and compare accuracy.{" "}
          <strong>LSTM</strong> reads sequences to spot streaks.{" "}
          <strong>XGBoost</strong> and <strong>LightGBM</strong> use statistical summaries.{" "}
          <strong>Linear (Ridge)</strong> is the simplest baseline.
        </p>
      </PageIntro>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-4" data-tour="train-controls">
          <TrainControls
            onTrain={handleTrain}
            isTraining={status?.is_training || false}
          />

          {/* Extra model checkboxes */}
          <div className="bg-mlb-card border border-mlb-border rounded-xl p-4 space-y-3">
            <p className="text-xs font-semibold text-mlb-muted uppercase tracking-wider">
              Additional Models
            </p>
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={trainLightGBM}
                onChange={(e) => setTrainLightGBM(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-mlb-text">
                <span className="text-mlb-green font-semibold">LightGBM</span>
                <span className="text-mlb-muted ml-1">— fast boosting baseline</span>
              </span>
            </label>
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={trainLinear}
                onChange={(e) => setTrainLinear(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-mlb-text">
                <span className="text-yellow-400 font-semibold">Linear (Ridge)</span>
                <span className="text-mlb-muted ml-1">— simplest baseline</span>
              </span>
            </label>
          </div>

          {status && <TrainProgress status={status} />}
        </div>

        {/* Metrics */}
        <div className="lg:col-span-2 space-y-4" data-tour="metrics">

          {/* Model comparison table */}
          {comparisonRows.length > 0 && (
            <div className="bg-mlb-card border border-mlb-border rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-mlb-border">
                <p className="text-xs font-semibold text-mlb-muted uppercase tracking-wider">
                  Model Comparison
                </p>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-mlb-border">
                    <th className="text-left text-xs text-mlb-muted px-4 py-2">Model</th>
                    <th className="text-right text-xs text-mlb-muted px-4 py-2">
                      MSE<InfoTooltip term="mse" />
                    </th>
                    <th className="text-right text-xs text-mlb-muted px-4 py-2">
                      MAE<InfoTooltip term="mae" />
                    </th>
                    <th className="text-right text-xs text-mlb-muted px-4 py-2">R²</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonRows.map((row) => {
                    const isBest = row.mse !== undefined && row.mse === bestMSE;
                    return (
                      <tr
                        key={row.key}
                        className={`border-b border-mlb-border/50 ${isBest ? "bg-mlb-green/5" : ""}`}
                      >
                        <td className={`px-4 py-3 font-semibold ${MODEL_COLORS[row.key]}`}>
                          {row.label}
                          {isBest && (
                            <span className="ml-2 text-[10px] bg-mlb-green/20 text-mlb-green px-1.5 py-0.5 rounded-full">
                              BEST
                            </span>
                          )}
                        </td>
                        <td className="text-right px-4 py-3 font-mono text-mlb-text">
                          {row.mse !== undefined ? formatMetric(row.mse) : "—"}
                        </td>
                        <td className="text-right px-4 py-3 font-mono text-mlb-text">
                          {row.mae !== undefined ? formatMetric(row.mae) : "—"}
                        </td>
                        <td className={`text-right px-4 py-3 font-mono ${
                          (row.r2 ?? 0) > 0 ? "text-mlb-green" : "text-mlb-red"
                        }`}>
                          {row.r2 !== undefined ? formatMetric(row.r2, 3) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Summary metric cards */}
          {modelKeys.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {modelKeys.map((k) => {
                const m = (metrics as Record<string, Record<string, number>>)?.[k];
                return (
                  <MetricCard
                    key={k}
                    label={
                      <>
                        <span className={MODEL_COLORS[k]}>{MODEL_LABELS[k]}</span>{" "}
                        MSE<InfoTooltip term="mse" />
                      </>
                    }
                    value={formatMetric(m?.mse)}
                    delta={`R² = ${formatMetric(m?.r2, 3)}`}
                    deltaType={(m?.r2 ?? 0) > 0 ? "positive" : "negative"}
                  />
                );
              })}
            </div>
          )}

          {/* Training Curves */}
          {curves && Object.values(curves).some(Boolean) && (
            <TrainingCurvesChart curves={curves} />
          )}

          {/* Per-Target MSE */}
          {perTargetData.length > 0 && (
            <BarComparison data={perTargetData} title="Per-Target MSE by Model" />
          )}

          {/* CV Results */}
          {evaluation?.cv_results && Object.keys(evaluation.cv_results).length > 0 && (
            <div className="bg-mlb-card border border-mlb-border rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-mlb-border">
                <p className="text-xs font-semibold text-mlb-muted uppercase tracking-wider">
                  5-Fold Time-Series Cross-Validation
                </p>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-mlb-border text-[11px] text-mlb-muted">
                    <th className="text-left px-4 py-2">Model</th>
                    <th className="text-right px-4 py-2">Mean MSE</th>
                    <th className="text-right px-4 py-2">Std MSE</th>
                    <th className="text-right px-4 py-2">Mean MAE</th>
                    <th className="text-right px-4 py-2">Folds</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(evaluation.cv_results as Record<string, Record<string, number>>).map(
                    ([k, cv]) => (
                      <tr key={k} className="border-b border-mlb-border/50">
                        <td className={`px-4 py-3 font-semibold ${MODEL_COLORS[k] ?? "text-mlb-text"}`}>
                          {MODEL_LABELS[k] ?? k}
                        </td>
                        <td className="text-right px-4 py-3 font-mono text-mlb-text">
                          {cv.mean_mse?.toFixed(4)}
                        </td>
                        <td className="text-right px-4 py-3 font-mono text-mlb-muted text-xs">
                          ±{cv.std_mse?.toFixed(4)}
                        </td>
                        <td className="text-right px-4 py-3 font-mono text-mlb-text">
                          {cv.mean_mae?.toFixed(4)}
                        </td>
                        <td className="text-right px-4 py-3 text-mlb-muted text-xs">
                          {cv.n_folds}
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>
          )}

          {/* Does Model Beat Baseline? */}
          {evaluation?.comparison && Object.keys(evaluation.comparison).length > 0 && (
            <div className="bg-mlb-card border border-mlb-border rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-mlb-border">
                <p className="text-xs font-semibold text-mlb-muted uppercase tracking-wider">
                  Does Model Beat Baseline?
                </p>
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-mlb-border text-mlb-muted">
                    <th className="text-left px-4 py-2">Matchup</th>
                    <th className="text-right px-3 py-2">Model MSE</th>
                    <th className="text-right px-3 py-2">Baseline MSE</th>
                    <th className="text-right px-3 py-2">Δ%</th>
                    <th className="text-right px-3 py-2">p-value</th>
                    <th className="text-right px-4 py-2">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(
                    evaluation.comparison as Record<string, Record<string, unknown>>
                  )
                    .filter(([k]) => k.includes("vs_season_average"))
                    .map(([key, comp]) => {
                      const wins = comp.model_wins as boolean;
                      const sig = comp.significant as boolean;
                      const pVal = comp.p_value as number | null;
                      const imp = comp.improvement_pct as number;
                      const modelName = key.split("_vs_")[0];
                      return (
                        <tr key={key} className="border-b border-mlb-border/50">
                          <td className="px-4 py-2">
                            <span className={MODEL_COLORS[modelName] ?? "text-mlb-text"}>
                              {MODEL_LABELS[modelName] ?? modelName}
                            </span>
                            <span className="text-mlb-muted ml-1">vs Season Avg</span>
                          </td>
                          <td className="text-right px-3 py-2 font-mono text-mlb-text">
                            {(comp.model_mse as number)?.toFixed(4)}
                          </td>
                          <td className="text-right px-3 py-2 font-mono text-mlb-muted">
                            {(comp.baseline_mse as number)?.toFixed(4)}
                          </td>
                          <td className={`text-right px-3 py-2 font-mono font-semibold ${wins ? "text-mlb-green" : "text-mlb-red"}`}>
                            {wins ? "+" : ""}{imp?.toFixed(1)}%
                          </td>
                          <td className="text-right px-3 py-2 text-mlb-muted font-mono">
                            {pVal != null ? pVal.toFixed(3) : "—"}
                          </td>
                          <td className="text-right px-4 py-2">
                            {wins ? (
                              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${sig ? "bg-mlb-green/20 text-mlb-green" : "bg-yellow-400/20 text-yellow-400"}`}>
                                {sig ? "✓ Beats" : "Marginal"}
                              </span>
                            ) : (
                              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-mlb-red/20 text-mlb-red">
                                ✗ Loses
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
              <p className="text-[10px] text-mlb-muted px-4 py-2">
                "Beats" = statistically significant improvement (p&lt;0.05, Wilcoxon signed-rank test).
                "Marginal" = better MSE but not significant.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
