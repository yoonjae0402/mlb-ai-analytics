"use client";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { TARGET_DISPLAY_NAMES } from "@/lib/constants";
import ChartExplainer from "@/components/ui/ChartExplainer";

interface PredictionVsActualProps {
  prediction: number[];
  actual: number[];
  targetNames: string[];
}

export default function PredictionVsActual({
  prediction,
  actual,
  targetNames,
}: PredictionVsActualProps) {
  const data = targetNames.map((name, i) => ({
    name: TARGET_DISPLAY_NAMES[name] || name,
    Predicted: Number(prediction[i]?.toFixed(2)) || 0,
    Actual: Number(actual[i]?.toFixed(2)) || 0,
  }));

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-1">
        Predicted vs Actual
      </h3>
      <ChartExplainer>
        Blue bars show what the model predicted; green bars show what actually happened. Closer bars mean a more accurate prediction.
      </ChartExplainer>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3050" />
          <XAxis type="number" stroke="#8899aa" fontSize={11} />
          <YAxis
            dataKey="name"
            type="category"
            stroke="#8899aa"
            fontSize={11}
            width={55}
          />
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
          <Bar
            dataKey="Predicted"
            fill="#4895ef"
            radius={[0, 4, 4, 0]}
            barSize={12}
          />
          <Bar
            dataKey="Actual"
            fill="#2dc653"
            radius={[0, 4, 4, 0]}
            barSize={12}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
