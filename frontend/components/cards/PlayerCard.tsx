import type { Player } from "@/lib/api";
import { cn } from "@/lib/utils";

interface PlayerCardProps {
  player: Player;
  selected?: boolean;
  onClick?: () => void;
}

export default function PlayerCard({ player, selected, onClick }: PlayerCardProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "bg-mlb-card border rounded-xl p-4 cursor-pointer transition-colors",
        selected
          ? "border-mlb-blue bg-mlb-blue/5"
          : "border-mlb-border hover:border-mlb-blue/40"
      )}
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-mlb-surface flex items-center justify-center text-sm font-bold text-mlb-blue">
          {player.name
            .split(" ")
            .map((n) => n[0])
            .join("")
            .slice(0, 2)}
        </div>
        <div>
          <p className="text-sm font-semibold text-mlb-text">{player.name}</p>
          <p className="text-xs text-mlb-muted">
            {player.team} {player.position && `- ${player.position}`}
          </p>
        </div>
      </div>
    </div>
  );
}
