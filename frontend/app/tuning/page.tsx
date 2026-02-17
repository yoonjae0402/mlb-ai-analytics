"use client";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { startTuning, getTuningStatus } from "@/lib/api";
import type { TuneStatus } from "@/lib/api";
import { Sliders, Play, CheckCircle, Loader2, AlertCircle } from "lucide-react";
import PageIntro from "@/components/ui/PageIntro";

const MODEL_OPTIONS = [
  {
    value: "lstm",
    label: "LSTM",
    description: "Tunes hidden size, dropout, learning rate, batch size, layers",
    color: "text-mlb-blue",
  },
  {
    value: "xgboost",
    label: "XGBoost",
    description: "Tunes n_estimators, max_depth, learning rate, subsample, colsample",
    color: "text-mlb-red",
  },
];

const TRIAL_OPTIONS = [10, 25, 50, 100];

export default function TuningPage() {
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState("lstm");
  const [nTrials, setNTrials] = useState(25);

  const { data: status, isLoading: statusLoading } = useQuery<TuneStatus>({
    queryKey: ["tuneStatus"],
    queryFn: getTuningStatus,
    refetchInterval: 3000,
  });

  const startMutation = useMutation({
    mutationFn: () => startTuning(selectedModel, nTrials),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tuneStatus"] });
    },
  });

  const isTuning = status?.is_tuning ?? false;
  const progress =
    status && status.n_trials > 0
      ? Math.round((status.completed_trials / status.n_trials) * 100)
      : 0;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <PageIntro
        title="Hyperparameter Tuning"
        icon={<Sliders className="w-5 h-5" />}
        pageKey="tuning"
      >
        <p>
          Optuna automatically searches for the best model hyperparameters using
          Bayesian optimization. Trials run in the background — check back for results.
          Best params are applied on the next training run.
        </p>
      </PageIntro>

      {/* Config */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-5 space-y-5">
        <h2 className="text-sm font-semibold text-mlb-text">Configuration</h2>

        {/* Model selection */}
        <div>
          <p className="text-xs text-mlb-muted mb-2">Model to tune</p>
          <div className="grid grid-cols-2 gap-3">
            {MODEL_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setSelectedModel(opt.value)}
                disabled={isTuning}
                className={`text-left border rounded-xl p-4 transition-all ${
                  selectedModel === opt.value
                    ? "border-mlb-blue bg-mlb-blue/10"
                    : "border-mlb-border bg-mlb-surface hover:border-mlb-blue/50"
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <p className={`text-sm font-bold ${opt.color}`}>{opt.label}</p>
                <p className="text-[11px] text-mlb-muted mt-1">{opt.description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Trials */}
        <div>
          <p className="text-xs text-mlb-muted mb-2">Number of trials</p>
          <div className="flex gap-2">
            {TRIAL_OPTIONS.map((t) => (
              <button
                key={t}
                onClick={() => setNTrials(t)}
                disabled={isTuning}
                className={`px-4 py-2 rounded-lg text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  nTrials === t
                    ? "bg-mlb-blue text-white"
                    : "bg-mlb-surface text-mlb-muted hover:text-mlb-text"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <p className="text-[10px] text-mlb-muted mt-1">
            More trials = better results, longer runtime (~{Math.round(nTrials * 0.5)}–{nTrials * 2}min)
          </p>
        </div>

        {/* Start button */}
        <button
          onClick={() => startMutation.mutate()}
          disabled={isTuning || startMutation.isPending}
          className="flex items-center gap-2 px-5 py-2.5 bg-mlb-blue hover:bg-mlb-blue/80 text-white rounded-xl text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isTuning ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Tuning in progress...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Start Tuning
            </>
          )}
        </button>

        {startMutation.isError && (
          <div className="flex items-center gap-2 text-xs text-mlb-red">
            <AlertCircle className="w-3.5 h-3.5" />
            Failed to start tuning. Is a model trained first?
          </div>
        )}
      </div>

      {/* Status Panel */}
      {!statusLoading && status && (
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-mlb-text">Tuning Status</h2>
            {isTuning ? (
              <span className="flex items-center gap-1.5 text-xs text-yellow-400">
                <Loader2 className="w-3 h-3 animate-spin" />
                Running
              </span>
            ) : status.completed_trials > 0 ? (
              <span className="flex items-center gap-1.5 text-xs text-mlb-green">
                <CheckCircle className="w-3 h-3" />
                Complete
              </span>
            ) : (
              <span className="text-xs text-mlb-muted">Not started</span>
            )}
          </div>

          {/* Progress bar */}
          {(isTuning || status.completed_trials > 0) && (
            <div>
              <div className="flex justify-between text-xs text-mlb-muted mb-1">
                <span>
                  {status.completed_trials} / {status.n_trials} trials
                </span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 bg-mlb-surface rounded-full overflow-hidden">
                <div
                  className="h-full bg-mlb-blue rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Info grid */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-mlb-surface rounded-lg p-3 text-center">
              <p className="text-[10px] text-mlb-muted mb-1">Model</p>
              <p className="text-sm font-bold text-mlb-text uppercase">
                {status.model_type || "—"}
              </p>
            </div>
            <div className="bg-mlb-surface rounded-lg p-3 text-center">
              <p className="text-[10px] text-mlb-muted mb-1">Completed</p>
              <p className="text-sm font-bold text-mlb-text">
                {status.completed_trials}
              </p>
            </div>
            <div className="bg-mlb-surface rounded-lg p-3 text-center">
              <p className="text-[10px] text-mlb-muted mb-1">Best Score</p>
              <p className="text-sm font-bold text-mlb-green">
                {status.best_score != null
                  ? status.best_score.toFixed(4)
                  : "—"}
              </p>
            </div>
          </div>

          {/* Best params */}
          {status.best_params && Object.keys(status.best_params).length > 0 && (
            <div>
              <p className="text-xs font-semibold text-mlb-muted uppercase tracking-wider mb-2">
                Best Parameters Found
              </p>
              <div className="bg-mlb-surface rounded-xl p-3 space-y-1.5">
                {Object.entries(status.best_params).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-xs">
                    <span className="text-mlb-muted font-mono">{k}</span>
                    <span className="text-mlb-text font-mono font-semibold">
                      {typeof v === "number" && !Number.isInteger(v)
                        ? (v as number).toFixed(6)
                        : String(v)}
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-mlb-muted mt-2">
                These will be applied automatically on the next training run from the Model Comparison page.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
