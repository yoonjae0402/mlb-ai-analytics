interface MetricCardProps {
  label: React.ReactNode;
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
  className = "",
}: MetricCardProps) {
  const deltaColor =
    deltaType === "positive" ? "var(--color-primary)" :
    deltaType === "negative" ? "#f97316" :
    "var(--color-muted)";

  return (
    <div
      className={`rounded-xl p-4 ${className}`}
      style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
    >
      <p className="text-[10px] font-medium uppercase tracking-wider" style={{ color: "var(--color-muted)" }}>
        {label}
      </p>
      <p className="text-2xl font-bold mt-1" style={{ color: "var(--color-text)" }}>{value}</p>
      {delta && (
        <p className="text-xs mt-1" style={{ color: deltaColor }}>
          {delta}
        </p>
      )}
    </div>
  );
}
