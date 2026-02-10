import { getContextLevel, CONTEXT_COLORS, type ContextLevel } from "@/lib/stat-helpers";

interface ContextBadgeProps {
  stat: string;
  value: number;
  level?: ContextLevel;
}

export default function ContextBadge({ stat, value, level }: ContextBadgeProps) {
  const ctx = level ?? getContextLevel(stat, value);
  const { bg, text, label } = CONTEXT_COLORS[ctx];

  return (
    <span
      className={`inline-flex items-center text-[10px] font-semibold px-1.5 py-0.5 rounded-full ${bg} ${text}`}
    >
      {label}
    </span>
  );
}
