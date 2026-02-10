"use client";
import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { startTraining, getTrainCurves, getModelMetrics, type TrainConfig } from "@/lib/api";
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

export default function ModelsPage() {
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

  const trainMutation = useMutation({
    mutationFn: (config: TrainConfig) => startTraining(config),
  });

  const handleTrain = (config: TrainConfig) => {
    trainMutation.mutate(config);
  };

  // Build per-target comparison data
  const perTargetData = metrics
    ? ["hits", "home_runs", "rbi", "walks"].map((target) => ({
        name: target.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()),
        lstm: metrics.lstm?.per_target?.[target]?.mse ?? 0,
        xgboost: metrics.xgboost?.per_target?.[target]?.mse ?? 0,
      }))
    : [];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <PageIntro title="Train & Compare AI Models" icon={<Brain className="w-5 h-5" />} pageKey="models">
        <p>
          Train two different AI models on real MLB Statcast data. <strong>LSTM</strong> reads
          a player&apos;s last 10 games in sequence to spot streaks and trends. <strong>XGBoost</strong> uses
          statistical summaries (averages, trends) to find patterns. Compare their accuracy below.
        </p>
      </PageIntro>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-4" data-tour="train-controls">
          <TrainControls
            onTrain={handleTrain}
            isTraining={status?.is_training || false}
          />
          {status && <TrainProgress status={status} />}
        </div>

        {/* Metrics */}
        <div className="lg:col-span-2 space-y-4" data-tour="metrics">
          {metrics && (metrics.lstm || metrics.xgboost) && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {metrics.lstm && (
                  <>
                    <MetricCard
                      label={<>LSTM MSE<InfoTooltip term="mse" /></>}
                      value={formatMetric(metrics.lstm.mse)}
                      delta={`R² = ${formatMetric(metrics.lstm.r2, 3)}`}
                      deltaType={metrics.lstm.r2 > 0 ? "positive" : "negative"}
                    />
                    <MetricCard
                      label={<>LSTM MAE<InfoTooltip term="mae" /></>}
                      value={formatMetric(metrics.lstm.mae)}
                    />
                  </>
                )}
                {metrics.xgboost && (
                  <>
                    <MetricCard
                      label={<>XGBoost MSE<InfoTooltip term="mse" /></>}
                      value={formatMetric(metrics.xgboost.mse)}
                      delta={`R² = ${formatMetric(metrics.xgboost.r2, 3)}`}
                      deltaType={metrics.xgboost.r2 > 0 ? "positive" : "negative"}
                    />
                  </>
                )}
              </div>
            </>
          )}

          {/* Training Curves */}
          {curves && (curves.lstm || curves.xgboost) && (
            <TrainingCurvesChart curves={curves} />
          )}

          {/* Per-Target MSE */}
          {perTargetData.length > 0 && (
            <BarComparison data={perTargetData} title="Per-Target MSE Comparison" />
          )}
        </div>
      </div>
    </div>
  );
}
