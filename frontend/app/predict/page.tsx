"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getPlayer, type Player } from "@/lib/api";
import { usePrediction } from "@/hooks/usePrediction";
import PlayerSearch from "@/components/predict/PlayerSearch";
import PredictionResultView from "@/components/predict/PredictionResult";
import RadarChart from "@/components/charts/RadarChart";
import MetricCard from "@/components/cards/MetricCard";

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

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Search + Controls */}
        <div className="space-y-4">
          <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
            <h3 className="text-sm font-semibold text-mlb-text mb-3">
              Find a Player
            </h3>
            <PlayerSearch
              onSelect={setSelectedPlayer}
              selectedId={selectedPlayer?.id}
            />
          </div>

          {selectedPlayer && (
            <div className="bg-mlb-card border border-mlb-border rounded-xl p-4 space-y-3">
              <div>
                <label className="text-xs text-mlb-muted block mb-1">Model</label>
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
              <h3 className="text-sm font-semibold text-mlb-text mb-2">
                {playerDetail.player.name}
              </h3>
              <p className="text-xs text-mlb-muted">
                {playerDetail.player.team} | {playerDetail.player.position || "—"} |
                Bats: {playerDetail.player.bats || "—"}
              </p>
              {playerDetail.season_totals && (
                <div className="grid grid-cols-3 gap-2 mt-3">
                  <div className="text-center">
                    <p className="text-lg font-bold text-mlb-text">
                      {playerDetail.season_totals.batting_avg?.toFixed(3)}
                    </p>
                    <p className="text-[10px] text-mlb-muted">AVG</p>
                  </div>
                  <div className="text-center">
                    <p className="text-lg font-bold text-mlb-text">
                      {playerDetail.season_totals.home_runs}
                    </p>
                    <p className="text-[10px] text-mlb-muted">HR</p>
                  </div>
                  <div className="text-center">
                    <p className="text-lg font-bold text-mlb-text">
                      {playerDetail.season_totals.rbi}
                    </p>
                    <p className="text-[10px] text-mlb-muted">RBI</p>
                  </div>
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
