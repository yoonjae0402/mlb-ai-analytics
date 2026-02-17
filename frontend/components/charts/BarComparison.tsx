"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import ChartExplainer from "@/components/ui/ChartExplainer";

// Default model display config
const DEFAULT_MODEL_COLORS: Record<string, string> = {
  LSTM: "#e63946",
  XGBoost: "#4895ef",
  LightGBM: "#2dc653",
  Linear: "#f4b942",
  Ensemble: "#9b5de5",
};

interface BarComparisonProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: Record<string, any>[];
  title?: string;
  /** Optional explicit list of bar keys (uses auto-detect if omitted) */
  keys?: string[];
}

export default function BarComparison({
  data,
  title = "Per-Target MSE",
  keys,
}: BarComparisonProps) {
  // Auto-detect bar keys: all keys except "name"
  const barKeys =
    keys ??
    (data.length > 0 ? Object.keys(data[0]).filter((k) => k !== "name") : []);

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-1">{title}</h3>
      <ChartExplainer>
        Compares prediction accuracy across stats. Shorter bars = lower error = better predictions.
      </ChartExplainer>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3050" />
          <XAxis dataKey="name" stroke="#8899aa" fontSize={11} />
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
          {barKeys.map((key, i) => (
            <Bar
              key={key}
              dataKey={key}
              fill={
                DEFAULT_MODEL_COLORS[key] ??
                `hsl(${(i * 60) % 360}, 70%, 55%)`
              }
              name={key}
              radius={[4, 4, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
