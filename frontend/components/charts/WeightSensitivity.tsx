"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import ChartExplainer from "@/components/ui/ChartExplainer";

interface WeightSensitivityProps {
  sweep: { lstm_weight: number; mse: number }[];
}

export default function WeightSensitivity({ sweep }: WeightSensitivityProps) {
  const minMSE = Math.min(...sweep.map((s) => s.mse));
  const bestWeight = sweep.find((s) => s.mse === minMSE)?.lstm_weight || 0.5;

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-1">
        Ensemble Weight Sensitivity
      </h3>
      <ChartExplainer>
        Shows how ensemble error changes across different LSTM/XGBoost weight blends. The green line marks the optimal weight.
      </ChartExplainer>
      <p className="text-xs text-mlb-muted mb-4">
        Best LSTM weight: {bestWeight.toFixed(2)} (MSE: {minMSE.toFixed(4)})
      </p>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={sweep}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3050" />
          <XAxis
            dataKey="lstm_weight"
            stroke="#8899aa"
            fontSize={11}
            label={{ value: "LSTM Weight", position: "bottom", fill: "#8899aa", fontSize: 11 }}
          />
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
          <ReferenceLine
            x={bestWeight}
            stroke="#2dc653"
            strokeDasharray="3 3"
          />
          <Line
            type="monotone"
            dataKey="mse"
            stroke="#4895ef"
            strokeWidth={2}
            dot={false}
            name="MSE"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
