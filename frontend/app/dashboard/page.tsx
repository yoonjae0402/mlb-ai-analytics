"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useLiveGames } from "@/hooks/useLiveGames";
import { getDataStatus, getWinProbability, getLeaderboard } from "@/lib/api";
import type { GameData, LeaderboardEntry } from "@/lib/api";
import GameCard from "@/components/cards/GameCard";
import WinProbabilityChart from "@/components/charts/WinProbabilityChart";
import {
  Activity, Calendar, Clock, Database, AlertCircle, TrendingUp, Users, RefreshCw,
} from "lucide-react";

export default function DashboardPage() {
  const router = useRouter();
  const { data, isLoading, error, refetch } = useLiveGames();
  const [selectedGame, setSelectedGame] = useState<GameData | null>(null);

  const { data: dataStatus } = useQuery({
    queryKey: ["dataStatus"],
    queryFn: getDataStatus,
    refetchInterval: 60_000,
  });

  const { data: leaderboard } = useQuery({
    queryKey: ["leaderboard"],
    queryFn: () => getLeaderboard(8),
    refetchInterval: 120_000,
    retry: false,
  });

  const { data: wpData } = useQuery({
    queryKey: ["winProb", selectedGame?.game_id],
    queryFn: () => (selectedGame ? getWinProbability(selectedGame.game_id) : null),
    enabled: !!selectedGame,
    retry: false,
  });

  const games = data?.games || [];
  const mode = data?.mode || "off_day";

  return (
    <div className="max-w-7xl mx-auto space-y-5">

      {/* Status Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatusTile
          icon={<Database className="w-4 h-4" />}
          label="Players Tracked"
          value={dataStatus?.players_count?.toLocaleString() ?? "—"}
          sub={`${dataStatus?.seasons?.length ?? 0} seasons`}
        />
        <StatusTile
          icon={<Activity className="w-4 h-4" />}
          label="Stat Records"
          value={dataStatus?.stats_count?.toLocaleString() ?? "—"}
        />
        <StatusTile
          icon={<Clock className="w-4 h-4" />}
          label="Last Updated"
          value={dataStatus?.last_updated ?? "—"}
        />
        <StatusTile
          icon={<TrendingUp className="w-4 h-4" />}
          label="Today's Games"
          value={games.length > 0 ? `${games.length} game${games.length === 1 ? "" : "s"}` : mode === "off_day" ? "Off Day" : "—"}
          highlight={mode === "live"}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left: Games Column */}
        <div className="lg:col-span-2 space-y-4">
          <SectionHeader title="Today's Games" badge={mode === "live" ? "LIVE" : undefined}>
            <button
              onClick={() => refetch()}
              className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded transition-colors"
              style={{
                background: "var(--color-dark)",
                color: "var(--color-muted)",
                border: "1px solid var(--color-border)",
              }}
              onMouseEnter={e => (e.currentTarget.style.color = "var(--color-secondary)")}
              onMouseLeave={e => (e.currentTarget.style.color = "var(--color-muted)")}
            >
              <RefreshCw className="w-3 h-3" />
              Refresh
            </button>
          </SectionHeader>

          {/* Loading skeleton */}
          {isLoading && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="fg-card p-4">
                  <div className="skeleton h-4 w-20 rounded mb-3" />
                  <div className="skeleton h-6 w-12 rounded mb-2" />
                  <div className="skeleton h-6 w-12 rounded mb-3" />
                  <div className="skeleton h-2 w-full rounded" />
                </div>
              ))}
            </div>
          )}

          {/* Error */}
          {error && !isLoading && (
            <div
              className="rounded-xl p-6 text-center"
              style={{ background: "var(--color-card)", border: "1px solid rgba(249,115,22,0.3)" }}
            >
              <AlertCircle className="w-6 h-6 mx-auto mb-2" style={{ color: "#f97316" }} />
              <p className="text-sm font-medium" style={{ color: "#f97316" }}>Unable to fetch game data</p>
              <p className="text-xs mt-1" style={{ color: "var(--color-muted)" }}>
                Make sure the backend API is running on port 8000.
              </p>
            </div>
          )}

          {/* Off Day */}
          {mode === "off_day" && !isLoading && !error && (
            <div
              className="rounded-xl p-10 text-center"
              style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
            >
              <Calendar className="w-8 h-8 mx-auto mb-3" style={{ color: "var(--color-muted)" }} />
              <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--color-text)" }}>
                No Games Today
              </h3>
              <p className="text-xs" style={{ color: "var(--color-muted)" }}>
                Check the{" "}
                <a href="/dashboard/schedule" style={{ color: "var(--color-secondary)" }}>Schedule</a>{" "}
                for upcoming matchups.
              </p>
            </div>
          )}

          {/* Games grid */}
          {games.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {games.map((game) => (
                <GameCard
                  key={game.game_id}
                  game={game}
                  onClick={() => {
                    if (selectedGame?.game_id === game.game_id) {
                      setSelectedGame(null);
                    } else {
                      setSelectedGame(game);
                    }
                  }}
                />
              ))}
            </div>
          )}

          {/* Selected Game Detail */}
          {selectedGame && (
            <div
              className="rounded-xl overflow-hidden"
              style={{ border: "1px solid var(--color-border)", background: "var(--color-card)" }}
            >
              <div
                className="px-4 py-3 flex items-center justify-between"
                style={{ background: "var(--color-panel)", borderBottom: "1px solid var(--color-border)" }}
              >
                <h3 className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>
                  {selectedGame.away_name} @ {selectedGame.home_name}
                </h3>
                <button
                  onClick={() => router.push(`/dashboard/game/${selectedGame.game_id}`)}
                  className="text-xs px-3 py-1 rounded font-medium"
                  style={{ background: "var(--color-primary)", color: "#1a1a2e" }}
                >
                  Full Analysis →
                </button>
              </div>

              <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--color-muted)" }}>
                    Score
                  </div>
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center justify-between">
                      <span className="font-bold" style={{ color: "var(--color-text)" }}>{selectedGame.away_name}</span>
                      <span className="text-2xl font-bold" style={{ color: "var(--color-text)" }}>{selectedGame.away_score}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="font-bold" style={{ color: "var(--color-text)" }}>{selectedGame.home_name}</span>
                      <span className="text-2xl font-bold" style={{ color: "var(--color-text)" }}>{selectedGame.home_score}</span>
                    </div>
                  </div>

                  {wpData && (
                    <div
                      className="rounded-lg p-3"
                      style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)" }}
                    >
                      <div className="text-[10px] font-semibold uppercase mb-2" style={{ color: "var(--color-muted)" }}>
                        Projected Runs
                        <span
                          className="ml-1 cursor-help"
                          data-tooltip="Lineup wOBA × park factor × starter ERA adjustment via Pythagorean expectation. Not a guarantee."
                          style={{ borderBottom: "1px dotted var(--color-accent)" }}
                        > (?)</span>
                      </div>
                      <div className="flex justify-between text-sm font-bold">
                        <span style={{ color: "var(--color-secondary)" }}>
                          {selectedGame.away_abbrev}: {wpData.away?.projected_runs?.toFixed(1) ?? "—"}
                        </span>
                        <span style={{ color: "var(--color-primary)" }}>
                          {selectedGame.home_abbrev}: {wpData.home?.projected_runs?.toFixed(1) ?? "—"}
                        </span>
                      </div>
                      <div className="text-[10px] mt-1" style={{ color: "var(--color-subtle)" }}>
                        Confidence: {wpData.confidence != null ? `${(wpData.confidence * 100).toFixed(0)}%` : "—"} ·
                        {wpData.method ?? "Pythagorean"}
                      </div>
                    </div>
                  )}
                </div>

                <WinProbabilityChart
                  wpHistory={selectedGame.wp_history}
                  homeTeam={selectedGame.home_abbrev}
                  awayTeam={selectedGame.away_abbrev}
                  compact
                />
              </div>
            </div>
          )}
        </div>

        {/* Right: Leaderboard */}
        <div className="space-y-4">
          <SectionHeader title="Player Leaderboard" sub="Season stats">
            <a href="/dashboard/leaderboard" style={{ color: "var(--color-secondary)", fontSize: "12px" }}>
              View all →
            </a>
          </SectionHeader>

          <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
            {!leaderboard || leaderboard.length === 0 ? (
              <div className="p-6 text-center">
                <Users className="w-5 h-5 mx-auto mb-2" style={{ color: "var(--color-muted)" }} />
                <p className="text-xs" style={{ color: "var(--color-muted)" }}>No leaderboard data yet.</p>
              </div>
            ) : (
              <div>
                <div
                  className="grid grid-cols-12 px-3 py-2 text-[10px] font-semibold uppercase tracking-wider"
                  style={{ background: "var(--color-panel)", borderBottom: "1px solid var(--color-border)", color: "var(--color-muted)" }}
                >
                  <div className="col-span-1">#</div>
                  <div className="col-span-5">Player</div>
                  <div className="col-span-2 text-right">H</div>
                  <div className="col-span-2 text-right">HR</div>
                  <div className="col-span-2 text-right">Score</div>
                </div>
                {leaderboard.map((entry: LeaderboardEntry, idx: number) => (
                  <LeaderboardRow key={entry.player_id} entry={entry} idx={idx} />
                ))}
              </div>
            )}
          </div>

          <p className="text-[10px] text-center" suppressHydrationWarning style={{ color: "var(--color-subtle)" }}>
            Auto-refreshes · {new Date().toLocaleTimeString()}
          </p>
        </div>
      </div>
    </div>
  );
}

function SectionHeader({
  title, sub, badge, children,
}: {
  title: string; sub?: string; badge?: string; children?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>{title}</h2>
        {badge && (
          <span
            className="flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full"
            style={{ background: "rgba(94,252,141,0.15)", color: "var(--color-primary)" }}
          >
            <span className="w-1.5 h-1.5 rounded-full live-dot" style={{ background: "var(--color-primary)" }} />
            {badge}
          </span>
        )}
        {sub && <span className="text-[10px]" style={{ color: "var(--color-subtle)" }}>{sub}</span>}
      </div>
      {children}
    </div>
  );
}

function StatusTile({
  icon, label, value, sub, highlight,
}: {
  icon: React.ReactNode; label: string; value: string; sub?: string; highlight?: boolean;
}) {
  return (
    <div
      className="rounded-xl p-3 flex items-center gap-3"
      style={{
        background: highlight ? "rgba(94,252,141,0.08)" : "var(--color-card)",
        border: `1px solid ${highlight ? "rgba(94,252,141,0.3)" : "var(--color-border)"}`,
      }}
    >
      <div style={{ color: highlight ? "var(--color-primary)" : "var(--color-muted)" }}>{icon}</div>
      <div className="min-w-0">
        <div className="text-[10px] font-medium" style={{ color: "var(--color-muted)" }}>{label}</div>
        <div className="text-sm font-bold truncate" style={{ color: highlight ? "var(--color-primary)" : "var(--color-text)" }}>
          {value}
        </div>
        {sub && <div className="text-[10px]" style={{ color: "var(--color-subtle)" }}>{sub}</div>}
      </div>
    </div>
  );
}

function LeaderboardRow({ entry, idx }: { entry: LeaderboardEntry; idx: number }) {
  const even = idx % 2 === 0;
  return (
    <a
      href={`/dashboard/player/${entry.player_id}`}
      className="grid grid-cols-12 items-center px-3 py-2 text-xs transition-colors"
      style={{
        background: even ? "transparent" : "rgba(131,119,209,0.06)",
        borderBottom: "1px solid var(--color-border)",
        color: "var(--color-text)",
        textDecoration: "none",
      }}
      onMouseEnter={e => (e.currentTarget.style.background = "rgba(94,252,141,0.07)")}
      onMouseLeave={e => (e.currentTarget.style.background = even ? "transparent" : "rgba(131,119,209,0.06)")}
    >
      <div className="col-span-1 font-bold" style={{ color: "var(--color-subtle)" }}>{entry.rank}</div>
      <div className="col-span-5">
        <div className="font-medium truncate">{entry.player_name}</div>
        <div className="text-[10px]" style={{ color: "var(--color-muted)" }}>{entry.team}</div>
      </div>
      <div className="col-span-2 text-right tabular-nums" style={{ color: "var(--color-secondary)" }}>
        {entry.predicted_hits?.toFixed(2) ?? "—"}
      </div>
      <div className="col-span-2 text-right tabular-nums" style={{ color: "var(--color-primary)" }}>
        {entry.predicted_hr?.toFixed(2) ?? "—"}
      </div>
      <div className="col-span-2 text-right tabular-nums font-medium" style={{ color: "var(--color-text)" }}>
        {entry.composite_score?.toFixed(1) ?? "—"}
      </div>
    </a>
  );
}
