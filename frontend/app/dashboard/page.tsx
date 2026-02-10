"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useLiveGames } from "@/hooks/useLiveGames";
import { getDataStatus, getSchedulerStatus, getBestBets } from "@/lib/api";
import type { DailyPrediction } from "@/lib/api";
import GameCard from "@/components/cards/GameCard";
import WinProbabilityChart from "@/components/charts/WinProbabilityChart";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import {
  Activity, Calendar, Clock, Database, Zap, TrendingUp, AlertCircle,
} from "lucide-react";

export default function DashboardPage() {
  const router = useRouter();
  const { data, isLoading, error } = useLiveGames();
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);

  const { data: dataStatus } = useQuery({
    queryKey: ["dataStatus"],
    queryFn: getDataStatus,
    refetchInterval: 60_000,
  });

  const { data: schedulerStatus } = useQuery({
    queryKey: ["schedulerStatus"],
    queryFn: getSchedulerStatus,
    refetchInterval: 30_000,
    retry: false,
  });

  const { data: bestBets } = useQuery({
    queryKey: ["bestBets"],
    queryFn: () => getBestBets(5),
    refetchInterval: 60_000,
    retry: false,
  });

  const games = data?.games || [];
  const mode = data?.mode || "off_day";
  const selectedGame = games.find((g) => g.game_id === selectedGameId);

  return (
    <div className="max-w-7xl mx-auto space-y-5">
      {/* System Status Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatusCard
          icon={<Database className="w-4 h-4" />}
          label="Players"
          value={dataStatus?.players_count?.toLocaleString() ?? "—"}
        />
        <StatusCard
          icon={<Activity className="w-4 h-4" />}
          label="Stat Records"
          value={dataStatus?.stats_count?.toLocaleString() ?? "—"}
        />
        <StatusCard
          icon={<Clock className="w-4 h-4" />}
          label="Last Updated"
          value={dataStatus?.last_updated ?? "—"}
        />
        <StatusCard
          icon={<Zap className="w-4 h-4" />}
          label="Next Prediction"
          value={schedulerStatus?.next_run ?? "4:00 AM ET"}
          status={schedulerStatus?.status}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left: Games */}
        <div className="lg:col-span-2 space-y-4">
          {/* Games Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-mlb-text">
                Today&apos;s Games
              </h2>
              {mode === "live" && (
                <span className="flex items-center gap-1 text-[10px] text-mlb-red font-semibold bg-mlb-red/10 px-2 py-0.5 rounded-full">
                  <span className="w-1.5 h-1.5 bg-mlb-red rounded-full animate-pulse" />
                  LIVE
                </span>
              )}
            </div>
            <span className="text-[10px] text-mlb-muted">Auto-refresh 30s</span>
          </div>

          {isLoading && (
            <div className="text-center py-8">
              <Activity className="w-6 h-6 text-mlb-muted animate-spin mx-auto" />
              <p className="text-xs text-mlb-muted mt-2">Loading games...</p>
            </div>
          )}

          {error && (
            <div className="bg-mlb-card border border-mlb-red/30 rounded-lg p-4 text-center">
              <p className="text-xs text-mlb-red">
                Unable to fetch game data. Make sure the backend is running.
              </p>
            </div>
          )}

          {mode === "off_day" && !isLoading && (
            <div className="bg-mlb-card border border-mlb-border rounded-lg p-8 text-center">
              <Calendar className="w-8 h-8 text-mlb-muted mx-auto mb-3" />
              <h3 className="text-sm font-semibold text-mlb-text">Off Day</h3>
              <p className="text-xs text-mlb-muted mt-1">
                No MLB games scheduled for today.
              </p>
            </div>
          )}

          {games.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5">
              {games.map((game) => (
                <GameCard
                  key={game.game_id}
                  game={game}
                  onClick={() => {
                    setSelectedGameId(game.game_id);
                    router.push(`/dashboard/game/${game.game_id}`);
                  }}
                />
              ))}
            </div>
          )}

          {/* Selected Game Detail */}
          {selectedGame && (
            <div className="bg-mlb-card border border-mlb-border rounded-lg p-4 space-y-3">
              <h3 className="text-sm font-semibold text-mlb-text">
                {selectedGame.away_name} @ {selectedGame.home_name}
              </h3>
              <div className="grid grid-cols-2 gap-3 text-center">
                <div>
                  <p className="text-2xl font-bold text-mlb-text">{selectedGame.away_score}</p>
                  <p className="text-[10px] text-mlb-muted">{selectedGame.away_abbrev}</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-mlb-text">{selectedGame.home_score}</p>
                  <p className="text-[10px] text-mlb-muted">{selectedGame.home_abbrev}</p>
                </div>
              </div>
              <WinProbabilityChart
                wpHistory={selectedGame.wp_history}
                homeTeam={selectedGame.home_abbrev}
                awayTeam={selectedGame.away_abbrev}
              />
            </div>
          )}
        </div>

        {/* Right: Best Bets / Benefit of the Doubt */}
        <div className="space-y-4">
          <div className="bg-mlb-card border border-mlb-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="w-4 h-4 text-mlb-red" />
              <h3 className="text-sm font-semibold text-mlb-text">
                Benefit of the Doubt
              </h3>
            </div>
            <p className="text-[10px] text-mlb-muted mb-3">
              Top model predictions where AI sees the most upside
            </p>

            {bestBets?.predictions && bestBets.predictions.length > 0 ? (
              <div className="space-y-2.5">
                {bestBets.predictions.map((pred: DailyPrediction, i: number) => (
                  <BestBetCard key={pred.player_id} prediction={pred} rank={i + 1} />
                ))}
              </div>
            ) : (
              <div className="text-center py-6">
                <AlertCircle className="w-5 h-5 text-mlb-muted mx-auto mb-2" />
                <p className="text-[10px] text-mlb-muted">
                  No predictions available yet. Run the prediction pipeline first.
                </p>
              </div>
            )}

            {bestBets?.last_updated && (
              <p className="text-[10px] text-mlb-muted mt-3 text-right">
                Updated: {new Date(bestBets.last_updated).toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusCard({
  icon,
  label,
  value,
  status,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  status?: string;
}) {
  return (
    <div className="bg-mlb-card border border-mlb-border rounded-lg p-3 flex items-center gap-3">
      <div className="text-mlb-muted">{icon}</div>
      <div>
        <p className="text-[10px] text-mlb-muted">{label}</p>
        <p className="text-sm font-semibold text-mlb-text">{value}</p>
        {status && (
          <p className={`text-[10px] ${status === "error" ? "text-mlb-red" : "text-green-400"}`}>
            {status}
          </p>
        )}
      </div>
    </div>
  );
}

function BestBetCard({ prediction, rank }: { prediction: DailyPrediction; rank: number }) {
  return (
    <div className="flex items-center gap-2.5 bg-mlb-surface/50 rounded-md p-2">
      <span className="text-xs font-bold text-mlb-muted w-5 text-center">#{rank}</span>
      <PlayerHeadshot url={prediction.headshot_url} name={prediction.player_name} size="sm" />
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-mlb-text truncate">{prediction.player_name}</p>
        <p className="text-[10px] text-mlb-muted">{prediction.team}</p>
      </div>
      <div className="text-right">
        <div className="flex gap-2 text-[10px]">
          <span className="text-mlb-text">
            <span className="text-mlb-muted">H:</span> {prediction.predicted_hits.toFixed(1)}
          </span>
          <span className="text-mlb-red font-medium">
            <span className="text-mlb-muted">HR:</span> {prediction.predicted_hr.toFixed(2)}
          </span>
        </div>
        <div className="flex gap-2 text-[10px]">
          <span className="text-mlb-text">
            <span className="text-mlb-muted">RBI:</span> {prediction.predicted_rbi.toFixed(1)}
          </span>
          <span className="text-mlb-text">
            <span className="text-mlb-muted">BB:</span> {prediction.predicted_walks.toFixed(1)}
          </span>
        </div>
      </div>
    </div>
  );
}
