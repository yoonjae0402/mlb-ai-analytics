"use client";
import { useState } from "react";
import type { TrainConfig } from "@/lib/api";
import InfoTooltip from "@/components/ui/InfoTooltip";

const FIELD_GLOSSARY: Record<string, string> = {
  epochs: "epochs",
  lr: "learning_rate",
  hidden_size: "hidden_size",
  batch_size: "batch_size",
  n_estimators: "xgboost",
  max_depth: "xgboost",
};

interface TrainControlsProps {
  onTrain: (config: TrainConfig) => void;
  isTraining: boolean;
}

export default function TrainControls({ onTrain, isTraining }: TrainControlsProps) {
  const [config, setConfig] = useState<TrainConfig>({
    epochs: 30,
    lr: 0.001,
    hidden_size: 64,
    batch_size: 32,
    n_estimators: 200,
    max_depth: 6,
    xgb_lr: 0.1,
    seasons: [2023, 2024],
  });

  const fields = [
    { key: "epochs", label: "Epochs", type: "number", min: 5, max: 200, step: 5 },
    { key: "lr", label: "Learning Rate", type: "number", min: 0.0001, max: 0.01, step: 0.0001 },
    { key: "hidden_size", label: "Hidden Size", type: "select", options: [32, 64, 128, 256] },
    { key: "batch_size", label: "Batch Size", type: "select", options: [16, 32, 64] },
    { key: "n_estimators", label: "XGB Estimators", type: "number", min: 50, max: 500, step: 50 },
    { key: "max_depth", label: "XGB Max Depth", type: "number", min: 3, max: 10, step: 1 },
  ] as const;

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-4">
        Training Configuration
      </h3>
      <div className="grid grid-cols-2 gap-3">
        {fields.map((field) => (
          <div key={field.key}>
            <label className="text-xs text-mlb-muted block mb-1">
              {field.label}
              {FIELD_GLOSSARY[field.key] && <InfoTooltip term={FIELD_GLOSSARY[field.key]} />}
            </label>
            {field.type === "select" ? (
              <select
                value={(config as any)[field.key]}
                onChange={(e) =>
                  setConfig({ ...config, [field.key]: Number(e.target.value) })
                }
                className="w-full bg-mlb-surface border border-mlb-border rounded-lg px-3 py-1.5 text-sm text-mlb-text"
              >
                {field.options.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                value={(config as any)[field.key]}
                min={field.min}
                max={field.max}
                step={field.step}
                onChange={(e) =>
                  setConfig({ ...config, [field.key]: Number(e.target.value) })
                }
                className="w-full bg-mlb-surface border border-mlb-border rounded-lg px-3 py-1.5 text-sm text-mlb-text"
              />
            )}
          </div>
        ))}
      </div>
      <button
        onClick={() => onTrain(config)}
        disabled={isTraining}
        className="mt-4 w-full bg-mlb-red hover:bg-mlb-red/80 disabled:opacity-50 text-white font-semibold py-2 px-4 rounded-lg transition-colors text-sm"
      >
        {isTraining ? "Training..." : "Train Models"}
      </button>
    </div>
  );
}
