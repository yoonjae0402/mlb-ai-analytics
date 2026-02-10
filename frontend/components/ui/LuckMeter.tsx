interface LuckMeterProps {
  actual: number;
  expected: number;
  label?: string;
  statName?: string;
}

export default function LuckMeter({ actual, expected, label, statName }: LuckMeterProps) {
  const diff = actual - expected;
  const absDiff = Math.abs(diff);
  const isLucky = diff > 0.01;
  const isUnlucky = diff < -0.01;

  const luckLabel = isLucky ? "Lucky" : isUnlucky ? "Unlucky" : "Fair";
  const luckColor = isLucky
    ? "text-mlb-green"
    : isUnlucky
    ? "text-mlb-red"
    : "text-mlb-muted";
  const barColor = isLucky
    ? "bg-mlb-green"
    : isUnlucky
    ? "bg-mlb-red"
    : "bg-slate-400";

  // Map diff to a visual position (center = 50%, lucky goes right, unlucky goes left)
  const maxDiff = 0.05;
  const normalizedDiff = Math.max(-maxDiff, Math.min(maxDiff, diff));
  const barWidth = (Math.abs(normalizedDiff) / maxDiff) * 50;
  const barLeft = diff >= 0 ? 50 : 50 - barWidth;

  return (
    <div className="space-y-1">
      {(label || statName) && (
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-mlb-muted">{label || statName}</span>
          <span className={`font-semibold ${luckColor}`}>
            {luckLabel} ({diff >= 0 ? "+" : ""}{diff.toFixed(3)})
          </span>
        </div>
      )}
      <div className="relative h-2 bg-slate-700/50 rounded-full overflow-hidden">
        {/* Center marker */}
        <div className="absolute top-0 left-1/2 w-px h-full bg-slate-500" />
        {/* Luck bar */}
        <div
          className={`absolute top-0 h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ left: `${barLeft}%`, width: `${barWidth}%` }}
        />
      </div>
      <div className="flex justify-between text-[9px] text-mlb-muted">
        <span>Actual: {actual.toFixed(3)}</span>
        <span>Expected: {expected.toFixed(3)}</span>
      </div>
    </div>
  );
}
