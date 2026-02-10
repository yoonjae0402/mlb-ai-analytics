"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import type { TrainCurves } from "@/lib/api";
import ChartExplainer from "@/components/ui/ChartExplainer";

interface TrainingCurvesProps {
  curves: TrainCurves;
  model?: "lstm" | "xgboost" | "both";
}

export default function TrainingCurvesChart({
  curves,
  model = "both",
}: TrainingCurvesProps) {
  const data: { epoch: number; [key: string]: number }[] = [];
  const maxLen = Math.max(
    curves.lstm?.train_losses?.length || 0,
    curves.xgboost?.train_losses?.length || 0
  );

  for (let i = 0; i < maxLen; i++) {
    const point: any = { epoch: i + 1 };
    if (curves.lstm && (model === "lstm" || model === "both")) {
      point.lstm_train = curves.lstm.train_losses[i];
      point.lstm_val = curves.lstm.val_losses[i];
    }
    if (curves.xgboost && (model === "xgboost" || model === "both")) {
      point.xgb_train = curves.xgboost.train_losses[i];
      point.xgb_val = curves.xgboost.val_losses[i];
    }
    data.push(point);
  }

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-1">Training Curves</h3>
      <ChartExplainer>
        These curves show how each model&apos;s error decreases during training. Solid lines = training, dashed = validation. A growing gap may indicate overfitting.
      </ChartExplainer>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3050" />
          <XAxis dataKey="epoch" stroke="#8899aa" fontSize={11} />
          <YAxis stroke="#8899aa" fontSize={11} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#111d32",
              border: "1px solid #1e3050",
              borderRadius: 8,
              color: "#e8ecf1",
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {(model === "lstm" || model === "both") && (
            <>
              <Line
                type="monotone"
                dataKey="lstm_train"
                stroke="#e63946"
                strokeWidth={2}
                dot={false}
                name="LSTM Train"
              />
              <Line
                type="monotone"
                dataKey="lstm_val"
                stroke="#e63946"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="LSTM Val"
              />
            </>
          )}
          {(model === "xgboost" || model === "both") && (
            <>
              <Line
                type="monotone"
                dataKey="xgb_train"
                stroke="#4895ef"
                strokeWidth={2}
                dot={false}
                name="XGBoost Train"
              />
              <Line
                type="monotone"
                dataKey="xgb_val"
                stroke="#4895ef"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="XGBoost Val"
              />
            </>
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
