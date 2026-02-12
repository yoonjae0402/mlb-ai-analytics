"use client";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  getAccuracySummary,
  getCalibrationData,
  triggerBackfill,
} from "@/lib/api";
import type { AccuracySummary, CalibrationPoint } from "@/lib/api";
import {
  Activity,
  Target,
  TrendingUp,
  BarChart3,
  RefreshCw,
  AlertCircle,
  ChevronLeft,
} from "lucide-react";
import Link from "next/link";

export default function AccuracyDashboard() {
  const { data: summary, isLoading, refetch } = useQuery({
    queryKey: ["accuracySummary"],
    queryFn: getAccuracySummary,
  });

  const { data: calibration } = useQuery({
    queryKey: ["calibrationData"],
    queryFn: getCalibrationData,
  });

  const backfillMutation = useMutation({
    mutationFn: () => triggerBackfill(14),
    onSuccess: () => {
      setTimeout(() => refetch(), 3000);
    },
  });

  if (isLoading) {
    return (
      <div className="max-w-6xl mx-auto flex items-center justify-center py-20">
        <Activity className="w-6 h-6 text-mlb-muted animate-spin" />
        <span className="ml-2 text-sm text-mlb-muted">
          Loading accuracy data...
        </span>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-1 text-xs text-mlb-muted hover:text-mlb-text mb-2"
          >
            <ChevronLeft className="w-4 h-4" /> Dashboard
          </Link>
          <h1 className="text-xl font-bold text-mlb-text">
            Prediction Accuracy
          </h1>
          <p className="text-xs text-mlb-muted mt-1">
            How well do our predictions match actual game outcomes?
          </p>
        </div>
        <button
          onClick={() => backfillMutation.mutate()}
          disabled={backfillMutation.isPending}
          className="flex items-center gap-1.5 text-xs bg-mlb-blue/20 text-mlb-blue px-3 py-1.5 rounded-lg hover:bg-mlb-blue/30 transition-colors disabled:opacity-50"
        >
          <RefreshCw
            className={`w-3.5 h-3.5 ${backfillMutation.isPending ? "animate-spin" : ""}`}
          />
          {backfillMutation.isPending ? "Evaluating..." : "Evaluate Recent"}
        </button>
      </div>

      {!summary || summary.total_evaluated === 0 ? (
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-8 text-center">
          <AlertCircle className="w-8 h-8 text-mlb-muted mx-auto mb-3" />
          <h3 className="text-sm font-semibold text-mlb-text mb-1">
            No Evaluated Predictions Yet
          </h3>
          <p className="text-xs text-mlb-muted max-w-md mx-auto">
            Click &quot;Evaluate Recent&quot; to compare predictions against
            actual game results. This requires completed games with player
            stats in the database.
          </p>
        </div>
      ) : (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              label="Predictions Evaluated"
              value={summary.total_evaluated.toString()}
              icon={<Target className="w-4 h-4 text-mlb-blue" />}
            />
            <MetricCard
              label="Average MSE"
              value={summary.avg_mse?.toFixed(4) ?? "—"}
              icon={<BarChart3 className="w-4 h-4 text-mlb-red" />}
              subtitle="Lower is better"
            />
            <MetricCard
              label="Average MAE"
              value={summary.avg_mae?.toFixed(4) ?? "—"}
              icon={<TrendingUp className="w-4 h-4 text-emerald-400" />}
              subtitle="Avg absolute error"
            />
            <MetricCard
              label="Hit Rate (within 1)"
              value={
                summary.hit_rate != null
                  ? `${Math.round(summary.hit_rate * 100)}%`
                  : "—"
              }
              icon={<Target className="w-4 h-4 text-amber-400" />}
              subtitle="Predicted hits within 1"
            />
          </div>

          {/* Per-Stat Breakdown */}
          <div className="bg-mlb-card border border-mlb-border rounded-xl p-5">
            <h3 className="text-sm font-semibold text-mlb-text mb-4">
              Per-Stat Accuracy
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(summary.per_stat).map(([stat, metrics]) => (
                <div
                  key={stat}
                  className="bg-mlb-surface/30 rounded-lg p-3"
                >
                  <p className="text-[10px] font-semibold text-mlb-muted uppercase tracking-wider mb-2">
                    {stat.replace("_", " ")}
                  </p>
                  {Object.entries(metrics as Record<string, number>).map(
                    ([metric, value]) => (
                      <div
                        key={metric}
                        className="flex justify-between text-xs mb-1"
                      >
                        <span className="text-mlb-muted">{metric}</span>
                        <span className="text-mlb-text font-mono">
                          {typeof value === "number" ? value.toFixed(4) : "—"}
                        </span>
                      </div>
                    )
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Calibration Chart */}
          {calibration && calibration.length > 0 && (
            <div className="bg-mlb-card border border-mlb-border rounded-xl p-5">
              <h3 className="text-sm font-semibold text-mlb-text mb-4">
                Calibration Curve
              </h3>
              <p className="text-[10px] text-mlb-muted mb-4">
                If the model is well-calibrated, bars should roughly follow the
                diagonal — higher confidence should correspond to higher
                accuracy.
              </p>
              <div className="space-y-2">
                {calibration.map((point) => (
                  <CalibrationRow key={point.confidence_bin} point={point} />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function MetricCard({
  label,
  value,
  icon,
  subtitle,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  subtitle?: string;
}) {
  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">{icon}
        <span className="text-[10px] font-semibold text-mlb-muted uppercase tracking-wider">
          {label}
        </span>
      </div>
      <p className="text-xl font-bold text-mlb-text tabular-nums">{value}</p>
      {subtitle && (
        <p className="text-[10px] text-mlb-muted mt-1">{subtitle}</p>
      )}
    </div>
  );
}

function CalibrationRow({ point }: { point: CalibrationPoint }) {
  const predicted = Math.round(point.predicted_accuracy * 100);
  const actual =
    point.actual_accuracy != null
      ? Math.round(point.actual_accuracy * 100)
      : null;

  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-20 text-mlb-muted text-right tabular-nums">
        {predicted}% conf
      </span>
      <div className="flex-1 h-4 bg-mlb-surface/30 rounded relative overflow-hidden">
        {/* Perfect calibration line */}
        <div
          className="absolute top-0 bottom-0 border-r border-mlb-muted/30"
          style={{ left: `${predicted}%` }}
        />
        {/* Actual accuracy bar */}
        {actual != null && (
          <div
            className={`h-full rounded transition-all ${
              Math.abs(actual - predicted) <= 15
                ? "bg-emerald-500/60"
                : Math.abs(actual - predicted) <= 30
                ? "bg-amber-500/60"
                : "bg-mlb-red/60"
            }`}
            style={{ width: `${actual}%` }}
          />
        )}
      </div>
      <span className="w-16 text-mlb-text tabular-nums">
        {actual != null ? `${actual}%` : "—"}
      </span>
      <span className="w-12 text-mlb-muted tabular-nums text-right">
        n={point.n_predictions}
      </span>
    </div>
  );
}
