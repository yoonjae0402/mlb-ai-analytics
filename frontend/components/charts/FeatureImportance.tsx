"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import { FEATURE_DISPLAY_NAMES } from "@/lib/constants";
import ChartExplainer from "@/components/ui/ChartExplainer";

interface FeatureImportanceProps {
  importance: number[];
  featureNames: string[];
}

export default function FeatureImportance({
  importance,
  featureNames,
}: FeatureImportanceProps) {
  const data = featureNames
    .map((name, i) => ({
      name: FEATURE_DISPLAY_NAMES[name] || name,
      importance: importance[i],
    }))
    .sort((a, b) => b.importance - a.importance);

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-1">
        Feature Importance (Gradient-based)
      </h3>
      <ChartExplainer>
        Shows which stats most influenced the prediction. Taller bars = bigger impact on the model&apos;s output.
      </ChartExplainer>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3050" />
          <XAxis type="number" stroke="#8899aa" fontSize={11} />
          <YAxis
            dataKey="name"
            type="category"
            stroke="#8899aa"
            fontSize={11}
            width={75}
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
          <Bar dataKey="importance" fill="#e63946" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
