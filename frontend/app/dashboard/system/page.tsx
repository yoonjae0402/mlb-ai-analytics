"use client";
import { useQuery } from "@tanstack/react-query";
import { getSystemHealth } from "@/lib/api";
import type { SystemHealth } from "@/lib/api";
import { Activity, Database, Cpu, Clock, CheckCircle, XCircle } from "lucide-react";

const MODEL_LABELS: Record<string, string> = {
  lstm: "LSTM",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  linear: "Linear (Ridge)",
};

const MODEL_COLORS: Record<string, string> = {
  lstm: "text-mlb-blue",
  xgboost: "text-mlb-red",
  lightgbm: "text-mlb-green",
  linear: "text-yellow-400",
};

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function StatCard({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4 flex items-start gap-3">
      <div className="text-mlb-muted mt-0.5">{icon}</div>
      <div>
        <p className="text-xs text-mlb-muted">{label}</p>
        <p className="text-lg font-bold text-mlb-text font-mono">{value}</p>
        {sub && <p className="text-[11px] text-mlb-muted mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

export default function SystemPage() {
  const { data: health, isLoading, error } = useQuery<SystemHealth>({
    queryKey: ["systemHealth"],
    queryFn: getSystemHealth,
    refetchInterval: 15000,
  });

  if (isLoading) {
    return (
      <div className="max-w-5xl mx-auto">
        <p className="text-mlb-muted text-sm text-center py-16">Loading system health...</p>
      </div>
    );
  }

  if (error || !health) {
    return (
      <div className="max-w-5xl mx-auto">
        <div className="bg-mlb-card border border-mlb-red/30 rounded-xl p-8 text-center">
          <XCircle className="w-6 h-6 text-mlb-red mx-auto mb-2" />
          <p className="text-mlb-muted text-sm">Could not reach backend health endpoint.</p>
        </div>
      </div>
    );
  }

  const allModels = ["lstm", "xgboost", "lightgbm", "linear"];

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Activity className="w-5 h-5 text-mlb-blue" />
        <div>
          <h1 className="text-lg font-bold text-mlb-text">System Health</h1>
          <p className="text-xs text-mlb-muted">
            API v{health.api_version} &middot; Updated {new Date(health.timestamp).toLocaleTimeString()}
          </p>
        </div>
      </div>

      {/* Top-level stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Clock className="w-4 h-4" />}
          label="Uptime"
          value={formatUptime(health.uptime_seconds)}
        />
        <StatCard
          icon={<Cpu className="w-4 h-4" />}
          label="Trained Models"
          value={health.trained_models.length}
          sub={health.trained_models.map((k) => MODEL_LABELS[k] ?? k).join(", ") || "None"}
        />
        <StatCard
          icon={<Database className="w-4 h-4" />}
          label="Players"
          value={health.db.player_count.toLocaleString()}
          sub={`${health.db.stat_rows.toLocaleString()} stat rows`}
        />
        <StatCard
          icon={<Activity className="w-4 h-4" />}
          label="Predictions"
          value={health.db.prediction_count.toLocaleString()}
          sub={`${health.db.game_count.toLocaleString()} games`}
        />
      </div>

      {/* Model Status Table */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-mlb-border">
          <p className="text-xs font-semibold text-mlb-muted uppercase tracking-wider">
            Model Status
          </p>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-mlb-border text-[11px] text-mlb-muted">
              <th className="text-left px-4 py-2">Model</th>
              <th className="text-center px-3 py-2">In Memory</th>
              <th className="text-right px-3 py-2">Val MSE</th>
              <th className="text-right px-3 py-2">Val R²</th>
              <th className="text-right px-3 py-2">Version</th>
              <th className="text-right px-4 py-2">Trained At</th>
            </tr>
          </thead>
          <tbody>
            {allModels.map((key) => {
              const inMemory = health.trained_models.includes(key);
              const ver = health.model_versions?.[key];
              const metrics = health.model_metrics?.[key];
              return (
                <tr key={key} className="border-b border-mlb-border/50">
                  <td className={`px-4 py-3 font-semibold ${MODEL_COLORS[key]}`}>
                    {MODEL_LABELS[key]}
                  </td>
                  <td className="px-3 py-3 text-center">
                    {inMemory ? (
                      <CheckCircle className="w-4 h-4 text-mlb-green mx-auto" />
                    ) : (
                      <XCircle className="w-4 h-4 text-mlb-muted mx-auto" />
                    )}
                  </td>
                  <td className="text-right px-3 py-3 font-mono text-mlb-text">
                    {metrics?.mse != null
                      ? metrics.mse.toFixed(4)
                      : ver?.val_mse != null
                      ? ver.val_mse.toFixed(4)
                      : "—"}
                  </td>
                  <td className={`text-right px-3 py-3 font-mono ${
                    ((metrics?.r2 ?? ver?.val_r2) ?? 0) > 0
                      ? "text-mlb-green"
                      : "text-mlb-muted"
                  }`}>
                    {metrics?.r2 != null
                      ? metrics.r2.toFixed(3)
                      : ver?.val_r2 != null
                      ? ver.val_r2.toFixed(3)
                      : "—"}
                  </td>
                  <td className="text-right px-3 py-3 text-mlb-muted font-mono text-xs">
                    {ver?.version ?? "—"}
                  </td>
                  <td className="text-right px-4 py-3 text-mlb-muted text-xs">
                    {ver?.trained_at
                      ? new Date(ver.trained_at).toLocaleDateString()
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Last retrain */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-4 flex items-center gap-3">
        <Clock className="w-4 h-4 text-mlb-muted shrink-0" />
        <div>
          <p className="text-xs text-mlb-muted">Last In-Process Retrain</p>
          <p className="text-sm font-mono text-mlb-text">
            {health.last_retrain_at
              ? new Date(health.last_retrain_at).toLocaleString()
              : "No retrain yet this session"}
          </p>
          <p className="text-[10px] text-mlb-muted mt-1">
            Daily retrains run automatically via <code>daily_retrain.py</code>. New models are promoted only if they beat the current champion by &gt;1% MSE.
          </p>
        </div>
      </div>
    </div>
  );
}
