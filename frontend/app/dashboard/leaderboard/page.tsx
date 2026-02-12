"use client";
import { useQuery } from "@tanstack/react-query";
import { getLeaderboard } from "@/lib/api";
import type { LeaderboardEntry } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import {
  Activity, Trophy, ChevronLeft,
} from "lucide-react";
import Link from "next/link";

export default function LeaderboardPage() {
  const { data: entries, isLoading } = useQuery({
    queryKey: ["leaderboard"],
    queryFn: () => getLeaderboard(25),
  });

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto flex items-center justify-center py-20">
        <Activity className="w-6 h-6 text-mlb-muted animate-spin" />
        <span className="ml-2 text-sm text-mlb-muted">Loading leaderboard...</span>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-5">
      <div>
        <Link href="/dashboard" className="inline-flex items-center gap-1 text-xs text-mlb-muted hover:text-mlb-text mb-2">
          <ChevronLeft className="w-4 h-4" /> Dashboard
        </Link>
        <div className="flex items-center gap-2">
          <Trophy className="w-5 h-5 text-amber-400" />
          <h1 className="text-xl font-bold text-mlb-text">Prediction Leaderboard</h1>
        </div>
        <p className="text-xs text-mlb-muted mt-1">
          Top predicted performers ranked by composite score (H + 4&times;HR + RBI)
        </p>
      </div>

      <div className="bg-mlb-card border border-mlb-border rounded-xl overflow-hidden">
        {/* Table Header */}
        <div className="grid grid-cols-[40px_1fr_60px_60px_60px_60px_80px] gap-2 px-4 py-3 border-b border-mlb-border text-[10px] font-semibold text-mlb-muted uppercase tracking-wider">
          <span className="text-center">#</span>
          <span>Player</span>
          <span className="text-center">H</span>
          <span className="text-center">HR</span>
          <span className="text-center">RBI</span>
          <span className="text-center">BB</span>
          <span className="text-center">Score</span>
        </div>

        {entries && entries.length > 0 ? (
          <div className="divide-y divide-mlb-border/50">
            {entries.map((entry) => (
              <LeaderboardRow key={entry.player_id} entry={entry} />
            ))}
          </div>
        ) : (
          <div className="p-8 text-center">
            <p className="text-xs text-mlb-muted">
              No predictions available. Generate baseline predictions first.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function LeaderboardRow({ entry }: { entry: LeaderboardEntry }) {
  const isTop3 = entry.rank <= 3;
  const rankColors = ["text-amber-400", "text-gray-300", "text-amber-600"];

  return (
    <div className="grid grid-cols-[40px_1fr_60px_60px_60px_60px_80px] gap-2 px-4 py-2.5 hover:bg-mlb-surface/30 transition-colors items-center">
      <span className={`text-center text-sm font-bold ${isTop3 ? rankColors[entry.rank - 1] : "text-mlb-muted"}`}>
        {entry.rank}
      </span>

      <div className="flex items-center gap-2 min-w-0">
        <PlayerHeadshot url={entry.headshot_url} name={entry.player_name} size="sm" />
        <div className="min-w-0">
          <p className="text-xs font-medium text-mlb-text truncate">{entry.player_name}</p>
          <p className="text-[10px] text-mlb-muted">{entry.team || "—"}</p>
        </div>
      </div>

      <p className="text-center text-xs font-semibold text-mlb-text tabular-nums">
        {entry.predicted_hits.toFixed(2)}
      </p>
      <p className="text-center text-xs font-semibold text-mlb-red tabular-nums">
        {entry.predicted_hr.toFixed(2)}
      </p>
      <p className="text-center text-xs font-semibold text-mlb-text tabular-nums">
        {entry.predicted_rbi.toFixed(2)}
      </p>
      <p className="text-center text-xs font-semibold text-mlb-text tabular-nums">
        {entry.predicted_walks.toFixed(2)}
      </p>
      <p className="text-center text-xs font-bold text-mlb-blue tabular-nums">
        {entry.composite_score.toFixed(2)}
      </p>
    </div>
  );
}
