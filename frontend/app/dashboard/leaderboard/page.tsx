"use client";
import { useQuery } from "@tanstack/react-query";
import { getLeaderboard } from "@/lib/api";
import type { LeaderboardEntry } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import Link from "next/link";
import { Trophy, AlertCircle } from "lucide-react";

const STAT_TIPS: Record<string, string> = {
  H: "Projected hits per game based on recent batting average and lineup projection",
  HR: "Projected home runs per game based on barrel rate, exit velocity, and park factors",
  RBI: "Projected runs batted in based on lineup position, runners on base context, and recent stats",
  BB: "Projected walks per game based on walk rate and opposing pitcher BB/9",
  Score: "Composite score = Hits + 4×HR + RBI. Higher is a stronger projected performer.",
};

export default function LeaderboardPage() {
  const { data: entries, isLoading, error } = useQuery({
    queryKey: ["leaderboard"],
    queryFn: () => getLeaderboard(25),
    refetchInterval: 120_000,
  });

  return (
    <div className="max-w-5xl mx-auto space-y-5">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Trophy className="w-5 h-5" style={{ color: "#f59e0b" }} />
          <h1 className="text-xl font-bold" style={{ color: "var(--color-text)" }}>Player Leaderboard</h1>
        </div>
        <p className="text-xs" style={{ color: "var(--color-muted)" }}>
          Top performers ranked by composite score (H + 4×HR + RBI) based on recent stats and projected performance.
          Hover column headers for explanations.
        </p>
      </div>

      {/* Loading skeleton */}
      {isLoading && (
        <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
          {Array.from({ length: 10 }).map((_, i) => (
            <div
              key={i}
              className="flex items-center gap-4 px-4 py-3"
              style={{ borderBottom: "1px solid var(--color-border)", background: i % 2 === 1 ? "rgba(131,119,209,0.06)" : "transparent" }}
            >
              <div className="skeleton w-6 h-4 rounded" />
              <div className="skeleton w-8 h-8 rounded-full" />
              <div className="flex-1">
                <div className="skeleton h-3 w-32 rounded mb-1" />
                <div className="skeleton h-2 w-16 rounded" />
              </div>
              {Array.from({ length: 5 }).map((_, j) => (
                <div key={j} className="skeleton h-4 w-12 rounded" />
              ))}
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
          <AlertCircle className="w-5 h-5 mx-auto mb-2" style={{ color: "#f97316" }} />
          <p className="text-sm" style={{ color: "#f97316" }}>Failed to load leaderboard.</p>
          <p className="text-xs mt-1" style={{ color: "var(--color-muted)" }}>Make sure the backend is running.</p>
        </div>
      )}

      {/* Table */}
      {!isLoading && !error && (
        <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
          {/* Header */}
          <div
            className="grid gap-2 px-4 py-2.5 text-[10px] font-semibold uppercase tracking-wider"
            style={{
              gridTemplateColumns: "40px 1fr 70px 70px 70px 70px 80px",
              background: "var(--color-panel)",
              borderBottom: "2px solid var(--color-primary)",
              color: "var(--color-muted)",
            }}
          >
            <span>#</span>
            <span>Player</span>
            {Object.keys(STAT_TIPS).map(stat => (
              <span
                key={stat}
                className="text-right stat-abbr cursor-help"
                data-tip={STAT_TIPS[stat]}
                style={{ borderBottom: "1px dotted var(--color-accent)" }}
              >
                {stat}
              </span>
            ))}
          </div>

          {entries && entries.length > 0 ? (
            <div>
              {entries.map((entry: LeaderboardEntry, idx: number) => (
                <LeaderboardRow key={entry.player_id} entry={entry} idx={idx} />
              ))}
            </div>
          ) : (
            <div className="p-10 text-center">
              <Trophy className="w-8 h-8 mx-auto mb-3" style={{ color: "var(--color-muted)" }} />
              <p className="text-sm" style={{ color: "var(--color-muted)" }}>
                No leaderboard data available.
              </p>
              <p className="text-xs mt-1" style={{ color: "var(--color-subtle)" }}>
                The leaderboard populates after the prediction pipeline runs.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Methodology note */}
      <div
        className="rounded-xl p-4 text-xs"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)", color: "var(--color-muted)" }}
      >
        <strong style={{ color: "var(--color-accent)" }}>How scores are calculated:</strong>{" "}
        Composite Score = Hits + 4×HR + RBI using projected per-game stats from our ML ensemble model.
        Stats reflect projected <em>per-game</em> performance, not season totals.
        See the <Link href="/architecture" style={{ color: "var(--color-secondary)" }}>Architecture page</Link> for full methodology.
      </div>
    </div>
  );
}

function LeaderboardRow({ entry, idx }: { entry: LeaderboardEntry; idx: number }) {
  const even = idx % 2 === 0;
  const rankColor =
    entry.rank === 1 ? "#f59e0b" :
    entry.rank === 2 ? "#94a3b8" :
    entry.rank === 3 ? "#cd7c2a" :
    "var(--color-subtle)";

  return (
    <Link
      href={`/dashboard/player/${entry.player_id}`}
      className="grid gap-2 items-center px-4 py-2.5 transition-colors"
      style={{
        gridTemplateColumns: "40px 1fr 70px 70px 70px 70px 80px",
        background: even ? "transparent" : "rgba(131,119,209,0.06)",
        borderBottom: "1px solid var(--color-border)",
        textDecoration: "none",
      }}
      onMouseEnter={e => (e.currentTarget.style.background = "rgba(94,252,141,0.06)")}
      onMouseLeave={e => (e.currentTarget.style.background = even ? "transparent" : "rgba(131,119,209,0.06)")}
    >
      {/* Rank */}
      <span className="text-sm font-bold text-center" style={{ color: rankColor }}>
        {entry.rank === 1 ? "🥇" : entry.rank === 2 ? "🥈" : entry.rank === 3 ? "🥉" : entry.rank}
      </span>

      {/* Player */}
      <div className="flex items-center gap-2 min-w-0">
        <PlayerHeadshot url={entry.headshot_url} name={entry.player_name} size="sm" />
        <div className="min-w-0">
          <p className="text-xs font-semibold truncate" style={{ color: "var(--color-text)" }}>
            {entry.player_name}
          </p>
          <p className="text-[10px]" style={{ color: "var(--color-muted)" }}>
            {entry.team || "—"}
          </p>
        </div>
      </div>

      {/* Stats */}
      <p className="text-right text-xs tabular-nums font-medium" style={{ color: "var(--color-secondary)" }}>
        {entry.predicted_hits?.toFixed(2) ?? "—"}
      </p>
      <p className="text-right text-xs tabular-nums font-bold" style={{ color: "var(--color-primary)" }}>
        {entry.predicted_hr?.toFixed(2) ?? "—"}
      </p>
      <p className="text-right text-xs tabular-nums" style={{ color: "var(--color-text)" }}>
        {entry.predicted_rbi?.toFixed(2) ?? "—"}
      </p>
      <p className="text-right text-xs tabular-nums" style={{ color: "var(--color-text)" }}>
        {entry.predicted_walks?.toFixed(2) ?? "—"}
      </p>
      <p className="text-right text-xs tabular-nums font-bold" style={{ color: "var(--color-accent)" }}>
        {entry.composite_score?.toFixed(2) ?? "—"}
      </p>
    </Link>
  );
}
