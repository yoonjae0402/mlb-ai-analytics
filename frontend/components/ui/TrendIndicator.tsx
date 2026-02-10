import { getTrend, type TrendDirection } from "@/lib/stat-helpers";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface TrendIndicatorProps {
  recent: number;
  season: number;
  direction?: TrendDirection;
}

const TREND_CONFIG: Record<TrendDirection, { icon: typeof TrendingUp; color: string; label: string }> = {
  up: { icon: TrendingUp, color: "text-mlb-green", label: "Trending up" },
  down: { icon: TrendingDown, color: "text-mlb-red", label: "Trending down" },
  flat: { icon: Minus, color: "text-mlb-muted", label: "Steady" },
};

export default function TrendIndicator({ recent, season, direction }: TrendIndicatorProps) {
  const trend = direction ?? getTrend(recent, season);
  const { icon: Icon, color, label } = TREND_CONFIG[trend];

  return (
    <span className={`inline-flex items-center ${color}`} title={label}>
      <Icon className="w-3.5 h-3.5" />
    </span>
  );
}
