import { cn } from "@/lib/utils";
import type { GameData } from "@/lib/api";
import InfoTooltip from "@/components/ui/InfoTooltip";

interface GameCardProps {
  game: GameData;
  onClick?: () => void;
}

export default function GameCard({ game, onClick }: GameCardProps) {
  const isLive = game.status === "In Progress" || game.status === "Live";
  const isFinal = game.status === "Final";

  return (
    <div
      onClick={onClick}
      className={cn(
        "bg-mlb-card border border-mlb-border rounded-xl p-4 cursor-pointer",
        "hover:border-mlb-blue/40 transition-colors"
      )}
    >
      <div className="flex items-center justify-between mb-3">
        <span
          className={cn(
            "text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full",
            isLive && "bg-mlb-red/20 text-mlb-red",
            isFinal && "bg-mlb-muted/20 text-mlb-muted",
            !isLive && !isFinal && "bg-mlb-blue/20 text-mlb-blue"
          )}
        >
          {isLive
            ? `${game.half} ${game.inning}`
            : game.status}
        </span>
        {game.venue && (
          <span className="text-[10px] text-mlb-muted truncate max-w-[100px]">
            {game.venue}
          </span>
        )}
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-mlb-text">
            {game.away_abbrev}
          </span>
          <span className="text-lg font-bold text-mlb-text">
            {game.away_score}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-mlb-text">
            {game.home_abbrev}
          </span>
          <span className="text-lg font-bold text-mlb-text">
            {game.home_score}
          </span>
        </div>
      </div>

      {(isLive || isFinal) && (
        <div className="mt-3">
          <div className="flex justify-between text-[10px] text-mlb-muted mb-1">
            <span>Win Prob<InfoTooltip term="win_probability" /></span>
            <span>{(game.home_win_prob * 100).toFixed(0)}%</span>
          </div>
          <div className="h-1.5 bg-mlb-surface rounded-full overflow-hidden">
            <div
              className="h-full bg-mlb-blue rounded-full transition-all duration-500"
              style={{ width: `${game.home_win_prob * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
