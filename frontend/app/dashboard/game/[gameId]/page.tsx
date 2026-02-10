"use client";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { getGamePredictions } from "@/lib/api";
import type { GamePlayerPrediction } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import ContextBadge from "@/components/ui/ContextBadge";
import PercentileBar from "@/components/ui/PercentileBar";
import StatTooltip from "@/components/ui/StatTooltip";
import {
  Activity, MapPin, Clock, Users, AlertCircle, ChevronLeft,
} from "lucide-react";
import Link from "next/link";

export default function GamePredictionPage() {
  const params = useParams();
  const gameId = params.gameId as string;

  const { data, isLoading, error } = useQuery({
    queryKey: ["gamePredictions", gameId],
    queryFn: () => getGamePredictions(gameId),
    enabled: !!gameId,
  });

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto flex items-center justify-center py-20">
        <Activity className="w-6 h-6 text-mlb-muted animate-spin" />
        <span className="ml-2 text-sm text-mlb-muted">Loading game data...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-7xl mx-auto py-10">
        <Link href="/dashboard" className="inline-flex items-center gap-1 text-xs text-mlb-muted hover:text-mlb-text mb-4">
          <ChevronLeft className="w-4 h-4" /> Back to Dashboard
        </Link>
        <div className="bg-mlb-card border border-mlb-red/30 rounded-lg p-8 text-center">
          <AlertCircle className="w-8 h-8 text-mlb-red mx-auto mb-3" />
          <h3 className="text-sm font-semibold text-mlb-text">Unable to load game</h3>
          <p className="text-xs text-mlb-muted mt-1">
            {error instanceof Error ? error.message : "Game data is not available."}
          </p>
        </div>
      </div>
    );
  }

  const { game, home_players, away_players } = data;

  return (
    <div className="max-w-7xl mx-auto space-y-5">
      {/* Back nav */}
      <Link href="/dashboard" className="inline-flex items-center gap-1 text-xs text-mlb-muted hover:text-mlb-text">
        <ChevronLeft className="w-4 h-4" /> Back to Dashboard
      </Link>

      {/* Game Header */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <span className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full ${
            game.status?.toLowerCase().includes("progress")
              ? "bg-mlb-red/20 text-mlb-red"
              : game.status?.toLowerCase().includes("final")
              ? "bg-mlb-muted/20 text-mlb-muted"
              : "bg-mlb-blue/20 text-mlb-blue"
          }`}>
            {game.status}
          </span>
          {game.venue && (
            <span className="flex items-center gap-1 text-[10px] text-mlb-muted">
              <MapPin className="w-3 h-3" />
              {game.venue}
            </span>
          )}
        </div>

        <div className="grid grid-cols-3 items-center gap-4">
          <div className="text-center">
            <p className="text-lg font-bold text-mlb-text">{game.away_team}</p>
            {game.away_probable_pitcher && game.away_probable_pitcher !== "TBD" && (
              <p className="text-[10px] text-mlb-muted mt-1">SP: {game.away_probable_pitcher}</p>
            )}
          </div>
          <div className="text-center">
            {game.away_score != null && game.home_score != null ? (
              <p className="text-3xl font-bold text-mlb-text">
                {game.away_score} - {game.home_score}
              </p>
            ) : (
              <div className="flex items-center justify-center gap-1 text-mlb-muted">
                <Clock className="w-4 h-4" />
                <span className="text-sm">
                  {game.game_datetime
                    ? new Date(game.game_datetime).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
                    : "TBD"}
                </span>
              </div>
            )}
            <p className="text-[10px] text-mlb-muted mt-1">
              {game.game_date || ""}
            </p>
          </div>
          <div className="text-center">
            <p className="text-lg font-bold text-mlb-text">{game.home_team}</p>
            {game.home_probable_pitcher && game.home_probable_pitcher !== "TBD" && (
              <p className="text-[10px] text-mlb-muted mt-1">SP: {game.home_probable_pitcher}</p>
            )}
          </div>
        </div>
      </div>

      {/* Player Predictions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Away Team */}
        <TeamRosterPanel
          teamName={game.away_team}
          players={away_players}
          label="Away"
        />

        {/* Home Team */}
        <TeamRosterPanel
          teamName={game.home_team}
          players={home_players}
          label="Home"
        />
      </div>

      {/* Legend */}
      <div className="bg-mlb-card border border-mlb-border rounded-lg p-4">
        <p className="text-[10px] font-semibold text-mlb-muted uppercase tracking-wider mb-2">How to read predictions</p>
        <div className="flex flex-wrap gap-4 text-[10px] text-mlb-muted">
          <span><strong className="text-mlb-text">H</strong> = Predicted Hits</span>
          <span><strong className="text-mlb-red">HR</strong> = Predicted Home Runs</span>
          <span><strong className="text-mlb-text">RBI</strong> = Predicted Runs Batted In</span>
          <span><strong className="text-mlb-text">BB</strong> = Predicted Walks</span>
          <span>Badges show how the prediction compares to league averages</span>
        </div>
      </div>
    </div>
  );
}

function TeamRosterPanel({
  teamName,
  players,
  label,
}: {
  teamName: string;
  players: GamePlayerPrediction[];
  label: string;
}) {
  const withPreds = players.filter((p) => p.has_prediction);
  const withoutPreds = players.filter((p) => !p.has_prediction);

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-mlb-border flex items-center gap-2">
        <Users className="w-4 h-4 text-mlb-blue" />
        <h3 className="text-sm font-semibold text-mlb-text">{teamName}</h3>
        <span className="text-[10px] text-mlb-muted ml-auto">{label}</span>
      </div>

      {players.length === 0 ? (
        <div className="p-6 text-center">
          <AlertCircle className="w-5 h-5 text-mlb-muted mx-auto mb-2" />
          <p className="text-xs text-mlb-muted">No players found in database for this team.</p>
        </div>
      ) : (
        <div className="divide-y divide-mlb-border/50">
          {/* Table Header */}
          <div className="grid grid-cols-[1fr_repeat(4,48px)] gap-1 px-4 py-2 text-[10px] font-semibold text-mlb-muted uppercase">
            <span>Player</span>
            <span className="text-center">H<StatTooltip stat="predicted_hits" /></span>
            <span className="text-center">HR<StatTooltip stat="predicted_hr" /></span>
            <span className="text-center">RBI<StatTooltip stat="predicted_rbi" /></span>
            <span className="text-center">BB<StatTooltip stat="predicted_walks" /></span>
          </div>

          {withPreds.map((p) => (
            <PlayerPredictionRow key={p.player_id} player={p} />
          ))}

          {withoutPreds.length > 0 && (
            <>
              <div className="px-4 py-2 bg-mlb-surface/30">
                <p className="text-[10px] text-mlb-muted">
                  {withoutPreds.length} player{withoutPreds.length > 1 ? "s" : ""} without predictions
                </p>
              </div>
              {withoutPreds.slice(0, 5).map((p) => (
                <PlayerPredictionRow key={p.player_id} player={p} />
              ))}
              {withoutPreds.length > 5 && (
                <div className="px-4 py-2 text-center text-[10px] text-mlb-muted">
                  +{withoutPreds.length - 5} more
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function PlayerPredictionRow({ player }: { player: GamePlayerPrediction }) {
  return (
    <div className="grid grid-cols-[1fr_repeat(4,48px)] gap-1 px-4 py-2 hover:bg-mlb-surface/30 transition-colors items-center">
      {/* Player Info */}
      <div className="flex items-center gap-2 min-w-0">
        <PlayerHeadshot url={player.headshot_url} name={player.name} size="sm" />
        <div className="min-w-0">
          <p className="text-xs font-medium text-mlb-text truncate">{player.name}</p>
          <p className="text-[10px] text-mlb-muted">{player.position || "—"}</p>
        </div>
      </div>

      {/* Stats */}
      {player.has_prediction ? (
        <>
          <PredStatCell stat="predicted_hits" value={player.predicted_hits!} />
          <PredStatCell stat="predicted_hr" value={player.predicted_hr!} isHighlight />
          <PredStatCell stat="predicted_rbi" value={player.predicted_rbi!} />
          <PredStatCell stat="predicted_walks" value={player.predicted_walks!} />
        </>
      ) : (
        <>
          <span className="text-center text-[10px] text-mlb-muted">—</span>
          <span className="text-center text-[10px] text-mlb-muted">—</span>
          <span className="text-center text-[10px] text-mlb-muted">—</span>
          <span className="text-center text-[10px] text-mlb-muted">—</span>
        </>
      )}
    </div>
  );
}

function PredStatCell({
  stat,
  value,
  isHighlight,
}: {
  stat: string;
  value: number;
  isHighlight?: boolean;
}) {
  return (
    <div className="text-center">
      <p className={`text-xs font-semibold tabular-nums ${isHighlight ? "text-mlb-red" : "text-mlb-text"}`}>
        {value.toFixed(2)}
      </p>
      <div className="mt-0.5">
        <PercentileBar stat={stat} value={value} showLabel={false} />
      </div>
    </div>
  );
}
