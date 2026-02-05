import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string | number;
  delta?: string;
  deltaType?: "positive" | "negative" | "neutral";
  className?: string;
}

export default function MetricCard({
  label,
  value,
  delta,
  deltaType = "neutral",
  className,
}: MetricCardProps) {
  return (
    <div
      className={cn(
        "bg-mlb-card border border-mlb-border rounded-xl p-4",
        className
      )}
    >
      <p className="text-xs text-mlb-muted uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold text-mlb-text mt-1">{value}</p>
      {delta && (
        <p
          className={cn(
            "text-xs mt-1",
            deltaType === "positive" && "text-mlb-green",
            deltaType === "negative" && "text-mlb-red",
            deltaType === "neutral" && "text-mlb-muted"
          )}
        >
          {delta}
        </p>
      )}
    </div>
  );
}
