interface StatRange {
  belowAvg: number;
  avg: number;
  good: number;
  label: string;
}

const STAT_RANGES: Record<string, StatRange> = {
  batting_avg: { belowAvg: 0.230, avg: 0.270, good: 0.300, label: "AVG" },
  home_runs: { belowAvg: 15, avg: 25, good: 35, label: "HR" },
  rbi: { belowAvg: 40, avg: 65, good: 90, label: "RBI" },
  obp: { belowAvg: 0.300, avg: 0.340, good: 0.380, label: "OBP" },
  slg: { belowAvg: 0.380, avg: 0.430, good: 0.500, label: "SLG" },
  walks: { belowAvg: 30, avg: 50, good: 70, label: "BB" },
  hits: { belowAvg: 100, avg: 140, good: 170, label: "H" },
};

function getRating(value: number, range: StatRange): { label: string; color: string } {
  if (value < range.belowAvg) return { label: "Below Avg", color: "#e63946" };
  if (value < range.avg) return { label: "Average", color: "#f59e0b" };
  if (value < range.good) return { label: "Good", color: "#2dc653" };
  return { label: "Elite", color: "#4895ef" };
}

function getPosition(value: number, range: StatRange): number {
  // Map value to 0-100 percentage. Elite ceiling = good * 1.3
  const max = range.good * 1.3;
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return pct;
}

interface StatGaugeProps {
  value: number;
  stat: string;
  label?: string;
}

export default function StatGauge({ value, stat, label }: StatGaugeProps) {
  const range = STAT_RANGES[stat];
  if (!range) return null;

  const rating = getRating(value, range);
  const position = getPosition(value, range);

  // Zone boundaries as percentages
  const max = range.good * 1.3;
  const z1 = (range.belowAvg / max) * 100;
  const z2 = (range.avg / max) * 100;
  const z3 = (range.good / max) * 100;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-mlb-muted">{label || range.label}</span>
        <span className="text-[10px] text-mlb-muted">
          <span className="font-mono text-mlb-text">
            {typeof value === "number" && value < 1 ? value.toFixed(3) : value}
          </span>{" "}
          <span style={{ color: rating.color }}>{rating.label}</span>
        </span>
      </div>
      <div className="relative h-2 rounded-full overflow-hidden bg-mlb-surface">
        {/* Color zones */}
        <div
          className="absolute inset-y-0 left-0 rounded-l-full"
          style={{ width: `${z1}%`, backgroundColor: "#e6394640" }}
        />
        <div
          className="absolute inset-y-0"
          style={{ left: `${z1}%`, width: `${z2 - z1}%`, backgroundColor: "#f59e0b40" }}
        />
        <div
          className="absolute inset-y-0"
          style={{ left: `${z2}%`, width: `${z3 - z2}%`, backgroundColor: "#2dc65340" }}
        />
        <div
          className="absolute inset-y-0 rounded-r-full"
          style={{ left: `${z3}%`, width: `${100 - z3}%`, backgroundColor: "#4895ef40" }}
        />
        {/* Marker */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full border-2 border-white shadow-sm"
          style={{
            left: `${position}%`,
            backgroundColor: rating.color,
            transform: `translate(-50%, -50%)`,
          }}
        />
      </div>
    </div>
  );
}
