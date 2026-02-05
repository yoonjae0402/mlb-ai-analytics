"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";

interface BarComparisonProps {
  data: { name: string; lstm: number; xgboost: number; ensemble?: number }[];
  title?: string;
}

export default function BarComparison({ data, title = "Per-Target MSE" }: BarComparisonProps) {
  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-4">{title}</h3>
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
          <Bar dataKey="lstm" fill="#e63946" name="LSTM" radius={[4, 4, 0, 0]} />
          <Bar dataKey="xgboost" fill="#4895ef" name="XGBoost" radius={[4, 4, 0, 0]} />
          {data.some((d) => d.ensemble !== undefined) && (
            <Bar dataKey="ensemble" fill="#2dc653" name="Ensemble" radius={[4, 4, 0, 0]} />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
