"use client";
import { useState } from "react";
import { useLiveGames } from "@/hooks/useLiveGames";
import GameCard from "@/components/cards/GameCard";
import WinProbabilityChart from "@/components/charts/WinProbabilityChart";
import { Activity, Calendar, Wifi } from "lucide-react";

export default function DashboardPage() {
  const { data, isLoading, error } = useLiveGames();
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);

  const games = data?.games || [];
  const mode = data?.mode || "off_day";
  const selectedGame = games.find((g) => g.game_id === selectedGameId);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Status Bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {mode === "live" && (
            <span className="flex items-center gap-1.5 text-xs text-mlb-red font-semibold">
              <span className="w-2 h-2 bg-mlb-red rounded-full animate-pulse" />
              LIVE
            </span>
          )}
          {mode === "schedule" && (
            <span className="flex items-center gap-1.5 text-xs text-mlb-blue">
              <Calendar className="w-3 h-3" />
              Today&apos;s Schedule
            </span>
          )}
          {mode === "off_day" && (
            <span className="flex items-center gap-1.5 text-xs text-mlb-muted">
              No Games Today
            </span>
          )}
        </div>
        <span className="text-xs text-mlb-muted">
          Auto-refreshing every 30s
        </span>
      </div>

      {isLoading && (
        <div className="text-center py-12">
          <Activity className="w-8 h-8 text-mlb-muted animate-spin mx-auto" />
          <p className="text-sm text-mlb-muted mt-2">Loading games...</p>
        </div>
      )}

      {error && (
        <div className="bg-mlb-card border border-mlb-red/30 rounded-xl p-6 text-center">
          <p className="text-sm text-mlb-red">
            Unable to fetch game data. Make sure the backend is running.
          </p>
        </div>
      )}

      {mode === "off_day" && !isLoading && (
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-12 text-center">
          <Calendar className="w-12 h-12 text-mlb-muted mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-mlb-text">Off Day</h3>
          <p className="text-sm text-mlb-muted mt-2">
            No MLB games scheduled for today. Check back tomorrow!
          </p>
        </div>
      )}

      {games.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Game Cards */}
          <div className="lg:col-span-2">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {games.map((game) => (
                <GameCard
                  key={game.game_id}
                  game={game}
                  onClick={() => setSelectedGameId(game.game_id)}
                />
              ))}
            </div>
          </div>

          {/* Selected Game Detail */}
          <div>
            {selectedGame ? (
              <div className="space-y-4">
                <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
                  <h3 className="text-sm font-semibold text-mlb-text mb-2">
                    {selectedGame.away_name} @ {selectedGame.home_name}
                  </h3>
                  <div className="grid grid-cols-2 gap-3 text-center">
                    <div>
                      <p className="text-2xl font-bold text-mlb-text">
                        {selectedGame.away_score}
                      </p>
                      <p className="text-xs text-mlb-muted">
                        {selectedGame.away_abbrev}
                      </p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-mlb-text">
                        {selectedGame.home_score}
                      </p>
                      <p className="text-xs text-mlb-muted">
                        {selectedGame.home_abbrev}
                      </p>
                    </div>
                  </div>
                </div>
                <WinProbabilityChart
                  wpHistory={selectedGame.wp_history}
                  homeTeam={selectedGame.home_abbrev}
                  awayTeam={selectedGame.away_abbrev}
                />
              </div>
            ) : (
              <div className="bg-mlb-card border border-mlb-border rounded-xl p-8 text-center">
                <p className="text-sm text-mlb-muted">
                  Select a game to view details
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
