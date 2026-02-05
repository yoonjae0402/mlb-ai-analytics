"use client";
import type { TrainStatus } from "@/lib/api";

interface TrainProgressProps {
  status: TrainStatus;
}

export default function TrainProgress({ status }: TrainProgressProps) {
  if (!status.is_training && status.progress === 0) return null;

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-mlb-text">
          Training Progress
        </h3>
        <span className="text-xs text-mlb-muted">
          {status.current_model} — Epoch {status.current_epoch}/{status.total_epochs}
        </span>
      </div>

      <div className="h-2 bg-mlb-surface rounded-full overflow-hidden mb-3">
        <div
          className="h-full bg-mlb-red rounded-full transition-all duration-300"
          style={{ width: `${status.progress}%` }}
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-mlb-muted">Train Loss</p>
          <p className="text-sm font-mono text-mlb-text">
            {status.train_loss.toFixed(4)}
          </p>
        </div>
        <div>
          <p className="text-xs text-mlb-muted">Val Loss</p>
          <p className="text-sm font-mono text-mlb-text">
            {status.val_loss.toFixed(4)}
          </p>
        </div>
      </div>

      {status.progress >= 100 && (
        <p className="text-xs text-mlb-green mt-2">Training complete!</p>
      )}
    </div>
  );
}
