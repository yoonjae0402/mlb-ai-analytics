"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getPlayer, type Player } from "@/lib/api";
import { usePrediction } from "@/hooks/usePrediction";
import PlayerSearch from "@/components/predict/PlayerSearch";
import PredictionResultView from "@/components/predict/PredictionResult";
import RadarChart from "@/components/charts/RadarChart";
import MetricCard from "@/components/cards/MetricCard";
import PageIntro from "@/components/ui/PageIntro";
import InfoTooltip from "@/components/ui/InfoTooltip";
import PlayerHeadshot from "@/components/ui/PlayerHeadshot";
import StatTrendline from "@/components/charts/StatTrendline";
import StatGauge from "@/components/ui/StatGauge";
import ContextBadge from "@/components/ui/ContextBadge";
import StatTooltip from "@/components/ui/StatTooltip";
import { Search } from "lucide-react";

export default function PredictPage() {
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
  const [modelType, setModelType] = useState("lstm");

  const { data: playerDetail } = useQuery({
    queryKey: ["player", selectedPlayer?.id],
    queryFn: () => getPlayer(selectedPlayer!.id),
    enabled: !!selectedPlayer,
  });

  const prediction = usePrediction();

  const handlePredict = () => {
    if (selectedPlayer) {
      prediction.mutate({ playerId: selectedPlayer.id, modelType });
    }
  };

  // Build sparkline data from recent_stats
  const buildTrendData = (key: string) => {
    if (!playerDetail?.recent_stats) return [];
    return playerDetail.recent_stats
      .filter((s) => (s as any)[key] !== undefined && (s as any)[key] !== null)
      .map((s) => ({
        game_date: s.game_date,
        value: (s as any)[key] as number,
      }));
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <PageIntro title="Predict a Player's Next Game" icon={<Search className="w-5 h-5" />} pageKey="predict">
        <p>
          Search for any MLB player, then let AI predict their next game performance.
          The model analyzes the player&apos;s last 10 games — including batting average,
          exit velocity, and other Statcast data — to forecast hits, home runs, RBI, and walks.
        </p>
      </PageIntro>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Search + Controls */}
        <div className="space-y-4">
          <div className="bg-mlb-card border border-mlb-border rounded-xl p-4" data-tour="player-search">
            <h3 className="text-sm font-semibold text-mlb-text mb-3">
              Find a Player
            </h3>
            <PlayerSearch
              onSelect={setSelectedPlayer}
              selectedId={selectedPlayer?.id}
            />
          </div>

          {selectedPlayer && (
            <div className="bg-mlb-card border border-mlb-border rounded-xl p-4 space-y-3" data-tour="predict-button">
              <div>
                <label className="text-xs text-mlb-muted block mb-1">
                  Model<InfoTooltip term="lstm" />
                </label>
                <div className="flex gap-2">
                  {["lstm", "xgboost", "ensemble"].map((m) => (
                    <button
                      key={m}
                      onClick={() => setModelType(m)}
                      className={`px-3 py-1.5 rounded-lg text-xs transition-colors ${
                        modelType === m
                          ? "bg-mlb-blue text-white"
                          : "bg-mlb-surface text-mlb-muted hover:text-mlb-text"
                      }`}
                    >
                      {m.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
              <button
                onClick={handlePredict}
                disabled={prediction.isPending}
                className="w-full bg-mlb-red hover:bg-mlb-red/80 disabled:opacity-50 text-white font-semibold py-2 px-4 rounded-lg transition-colors text-sm"
              >
                {prediction.isPending ? "Predicting..." : "Predict Next Game"}
              </button>
            </div>
          )}

          {/* Player Info */}
          {playerDetail && (
            <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
              <div className="flex items-center gap-3 mb-3">
                <PlayerHeadshot
                  mlbId={playerDetail.player.mlb_id}
                  name={playerDetail.player.name}
                  size="md"
                />
                <div>
                  <h3 className="text-sm font-semibold text-mlb-text">
                    {playerDetail.player.name}
                  </h3>
                  <p className="text-xs text-mlb-muted">
                    {playerDetail.player.team} | {playerDetail.player.position || "—"} |
                    Bats: {playerDetail.player.bats || "—"}
                  </p>
                </div>
              </div>
              {playerDetail.season_totals && (
                <>
                  <div className="grid grid-cols-3 gap-2 mb-3">
                    <div className="text-center">
                      <p className="text-lg font-bold text-mlb-text">
                        {playerDetail.season_totals.batting_avg?.toFixed(3)}
                      </p>
                      <div className="flex items-center justify-center gap-1">
                        <p className="text-[10px] text-mlb-muted">AVG<StatTooltip stat="batting_avg" /></p>
                        {playerDetail.season_totals.batting_avg != null && (
                          <ContextBadge stat="batting_avg" value={playerDetail.season_totals.batting_avg} />
                        )}
                      </div>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-bold text-mlb-text">
                        {playerDetail.season_totals.home_runs}
                      </p>
                      <div className="flex items-center justify-center gap-1">
                        <p className="text-[10px] text-mlb-muted">HR<StatTooltip stat="home_runs" /></p>
                        {playerDetail.season_totals.home_runs != null && (
                          <ContextBadge stat="home_runs" value={playerDetail.season_totals.home_runs} />
                        )}
                      </div>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-bold text-mlb-text">
                        {playerDetail.season_totals.rbi}
                      </p>
                      <div className="flex items-center justify-center gap-1">
                        <p className="text-[10px] text-mlb-muted">RBI<StatTooltip stat="rbi" /></p>
                        {playerDetail.season_totals.rbi != null && (
                          <ContextBadge stat="rbi" value={playerDetail.season_totals.rbi} />
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Stat Gauges */}
                  <div className="space-y-2 mb-3">
                    {playerDetail.season_totals.batting_avg != null && (
                      <StatGauge value={playerDetail.season_totals.batting_avg} stat="batting_avg" />
                    )}
                    {playerDetail.season_totals.home_runs != null && (
                      <StatGauge value={playerDetail.season_totals.home_runs} stat="home_runs" />
                    )}
                    {playerDetail.season_totals.rbi != null && (
                      <StatGauge value={playerDetail.season_totals.rbi} stat="rbi" />
                    )}
                  </div>
                </>
              )}

              {/* Stat Sparklines */}
              {playerDetail.recent_stats && playerDetail.recent_stats.length > 1 && (
                <div className="space-y-2 pt-3 border-t border-mlb-border">
                  <p className="text-[10px] text-mlb-muted uppercase tracking-wider mb-1">
                    Recent Trends
                  </p>
                  <StatTrendline
                    data={buildTrendData("batting_avg")}
                    label="Batting Avg"
                    color="#e63946"
                  />
                  <StatTrendline
                    data={buildTrendData("home_runs")}
                    label="Home Runs"
                    color="#4895ef"
                  />
                  <StatTrendline
                    data={buildTrendData("rbi")}
                    label="RBI"
                    color="#2dc653"
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {prediction.error && (
            <div className="bg-mlb-card border border-mlb-red/30 rounded-xl p-4">
              <p className="text-sm text-mlb-red">
                {(prediction.error as Error).message}
              </p>
            </div>
          )}

          {prediction.data && (
            <>
              <PredictionResultView result={prediction.data} />
              {prediction.data.last_features.length > 0 && (
                <RadarChart
                  playerFeatures={prediction.data.last_features}
                  featureNames={prediction.data.feature_names}
                />
              )}
            </>
          )}

          {!prediction.data && !prediction.error && (
            <div className="bg-mlb-card border border-mlb-border rounded-xl p-12 text-center">
              <p className="text-sm text-mlb-muted">
                Search for a player and click &quot;Predict Next Game&quot; to see predictions.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
