"use client";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import ChartExplainer from "@/components/ui/ChartExplainer";

interface WinProbabilityChartProps {
  wpHistory: number[];
  homeTeam: string;
  awayTeam: string;
}

export default function WinProbabilityChart({
  wpHistory,
  homeTeam,
  awayTeam,
}: WinProbabilityChartProps) {
  const data = wpHistory.map((wp, i) => ({
    inning: i === 0 ? "Pre" : `${i}`,
    home_wp: wp * 100,
    away_wp: (1 - wp) * 100,
  }));

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-1">
        Win Probability
      </h3>
      <ChartExplainer>
        Tracks the home team&apos;s win chance as the game progresses. Above 50% = home team favored.
      </ChartExplainer>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3050" />
          <XAxis dataKey="inning" stroke="#8899aa" fontSize={11} />
          <YAxis
            domain={[0, 100]}
            stroke="#8899aa"
            fontSize={11}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#111d32",
              border: "1px solid #1e3050",
              borderRadius: 8,
              color: "#e8ecf1",
              fontSize: 12,
            }}
            formatter={(val: number) => `${val.toFixed(1)}%`}
          />
          <ReferenceLine y={50} stroke="#8899aa" strokeDasharray="3 3" />
          <Area
            type="monotone"
            dataKey="home_wp"
            stroke="#4895ef"
            fill="#4895ef"
            fillOpacity={0.2}
            name={homeTeam}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
