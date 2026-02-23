"use client";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

interface WinProbabilityChartProps {
  wpHistory: number[];
  homeTeam: string;
  awayTeam: string;
  compact?: boolean;
}

interface TooltipPayload {
  value: number;
  name: string;
  color: string;
}

function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg p-3 text-xs shadow-xl"
      style={{
        background: "var(--color-deeper)",
        border: "1px solid var(--color-accent)",
        color: "var(--color-text)",
      }}
    >
      <div className="font-semibold mb-1" style={{ color: "var(--color-muted)" }}>
        {label === "Pre" ? "Pre-game" : `Inning ${label}`}
      </div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span style={{ color: p.color }}>{p.name}:</span>
          <span className="font-bold">{Number(p.value).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

export default function WinProbabilityChart({
  wpHistory,
  homeTeam,
  awayTeam,
  compact = false,
}: WinProbabilityChartProps) {
  if (!wpHistory || wpHistory.length === 0) {
    return (
      <div
        className="rounded-lg p-4 text-center"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
      >
        <p className="text-xs" style={{ color: "var(--color-muted)" }}>
          Win probability data not available
        </p>
      </div>
    );
  }

  const data = wpHistory.map((wp, i) => ({
    inning: i === 0 ? "Pre" : `${i}`,
    [homeTeam]: parseFloat((wp * 100).toFixed(1)),
    [awayTeam]: parseFloat(((1 - wp) * 100).toFixed(1)),
  }));

  const height = compact ? 140 : 220;

  return (
    <div
      className="rounded-xl p-4"
      style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
    >
      {!compact && (
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>
            Win Probability
          </h3>
          <span
            className="text-[10px] px-2 py-0.5 rounded"
            data-tooltip="Based on Pythagorean expectation using projected run totals from lineup and pitcher matchup analysis"
            style={{
              background: "rgba(131,119,209,0.15)",
              color: "var(--color-accent)",
              cursor: "help",
              borderBottom: "1px dotted var(--color-accent)",
            }}
          >
            Statistical Analysis
          </span>
        </div>
      )}

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -16 }}>
          <defs>
            <linearGradient id="homeGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#5efc8d" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#5efc8d" stopOpacity={0.03} />
            </linearGradient>
            <linearGradient id="awayGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8ef9f3" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#8ef9f3" stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(147,190,223,0.15)" />
          <XAxis
            dataKey="inning"
            stroke="rgba(147,190,223,0.4)"
            fontSize={10}
            tick={{ fill: "var(--color-subtle)" }}
          />
          <YAxis
            domain={[0, 100]}
            stroke="rgba(147,190,223,0.4)"
            fontSize={10}
            tick={{ fill: "var(--color-subtle)" }}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={50}
            stroke="rgba(147,190,223,0.5)"
            strokeDasharray="4 4"
            label={{ value: "50%", fill: "var(--color-subtle)", fontSize: 9, position: "right" }}
          />
          <Area
            type="monotone"
            dataKey={homeTeam}
            stroke="#5efc8d"
            strokeWidth={2}
            fill="url(#homeGrad)"
            dot={false}
            activeDot={{ r: 4, fill: "#5efc8d" }}
          />
          <Area
            type="monotone"
            dataKey={awayTeam}
            stroke="#8ef9f3"
            strokeWidth={2}
            fill="url(#awayGrad)"
            dot={false}
            activeDot={{ r: 4, fill: "#8ef9f3" }}
          />
          {!compact && (
            <Legend
              wrapperStyle={{ fontSize: 10, color: "var(--color-muted)" }}
              formatter={(value) => (
                <span style={{ color: "var(--color-muted)" }}>{value} (Home: {homeTeam})</span>
              )}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>

      {!compact && (
        <p className="text-[10px] mt-2" style={{ color: "var(--color-subtle)" }}>
          Win probability uses Pythagorean expectation based on projected lineup wOBA, park factors, and
          starting pitcher ERA. This is <em>statistical analysis</em>, not a guaranteed prediction.
        </p>
      )}
    </div>
  );
}
