"use client";
import { LineChart, Line, ResponsiveContainer, Tooltip } from "recharts";

interface StatTrendlineProps {
  data: { game_date: string; value: number }[];
  label: string;
  color?: string;
}

export default function StatTrendline({
  data,
  label,
  color = "#4895ef",
}: StatTrendlineProps) {
  if (!data || data.length < 2) return null;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-mlb-muted">{label}</span>
        <span className="text-[10px] font-mono text-mlb-text">
          {data[data.length - 1]?.value?.toFixed(3)}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={48}>
        <LineChart data={data}>
          <Tooltip
            contentStyle={{
              backgroundColor: "#111d32",
              border: "1px solid #1e3050",
              borderRadius: 6,
              color: "#e8ecf1",
              fontSize: 10,
              padding: "4px 8px",
            }}
            formatter={(val: number) => [val.toFixed(3), label]}
            labelFormatter={(l) => `Game: ${l}`}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 2, fill: color }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
