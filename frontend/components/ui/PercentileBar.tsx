import { getPercentile, getPercentileColor, getPercentileTextColor } from "@/lib/stat-helpers";

interface PercentileBarProps {
  stat: string;
  value: number;
  percentile?: number;
  showLabel?: boolean;
}

export default function PercentileBar({ stat, value, percentile, showLabel = true }: PercentileBarProps) {
  const pct = percentile ?? getPercentile(stat, value);
  const barColor = getPercentileColor(pct);
  const textColor = getPercentileTextColor(pct);

  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-2 bg-slate-700/50 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && (
        <span className={`text-[11px] font-bold tabular-nums w-6 text-right ${textColor}`}>
          {pct}
        </span>
      )}
    </div>
  );
}
