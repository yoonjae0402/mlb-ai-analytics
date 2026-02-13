"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { searchPitchers, getPitcherStats } from "@/lib/api";
import type { Player, PitcherStatsResult } from "@/lib/api";
import PlayerHeadshot from "@/components/ui/PlayerHeadshot";
import { TrendingUp, Search, Activity } from "lucide-react";

export default function PitchersPage() {
  const [query, setQuery] = useState("");
  const [selectedPitcher, setSelectedPitcher] = useState<Player | null>(null);

  const { data: pitchers, isLoading: searchLoading } = useQuery({
    queryKey: ["pitcherSearch", query],
    queryFn: () => searchPitchers(query, 20),
    enabled: query.length > 0,
  });

  const { data: defaultPitchers } = useQuery({
    queryKey: ["pitcherSearch", ""],
    queryFn: () => searchPitchers("", 20),
  });

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["pitcherStats", selectedPitcher?.id],
    queryFn: () => getPitcherStats(selectedPitcher!.id),
    enabled: !!selectedPitcher,
  });

  const displayList = query.length > 0 ? pitchers : defaultPitchers;

  return (
    <div className="max-w-7xl mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <TrendingUp className="w-5 h-5 text-mlb-red" />
        <div>
          <h1 className="text-lg font-bold text-mlb-text">Pitcher Stats</h1>
          <p className="text-xs text-mlb-muted">
            Search and view pitcher performance metrics
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Search Panel */}
        <div className="space-y-4">
          <div className="bg-mlb-card border border-mlb-border rounded-lg p-4">
            <div className="relative mb-3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-mlb-muted" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search pitchers..."
                className="w-full pl-9 pr-3 py-2 bg-mlb-surface border border-mlb-border rounded-lg text-sm text-mlb-text placeholder:text-mlb-muted focus:outline-none focus:border-mlb-blue"
              />
            </div>

            {searchLoading ? (
              <p className="text-xs text-mlb-muted text-center py-4">Searching...</p>
            ) : (
              <div className="space-y-1 max-h-[500px] overflow-y-auto">
                {(displayList || []).map((p) => (
                  <button
                    key={p.id}
                    onClick={() => setSelectedPitcher(p)}
                    className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-colors ${
                      selectedPitcher?.id === p.id
                        ? "bg-mlb-red/10 border border-mlb-red/30"
                        : "hover:bg-mlb-surface"
                    }`}
                  >
                    <PlayerHeadshot
                      mlbId={p.mlb_id}
                      name={p.name}
                      size="sm"
                    />
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-mlb-text truncate">
                        {p.name}
                      </p>
                      <p className="text-[10px] text-mlb-muted">
                        {p.team || "—"} &middot; {p.throws ? `Throws ${p.throws}` : "P"}
                      </p>
                    </div>
                  </button>
                ))}
                {displayList && displayList.length === 0 && (
                  <p className="text-xs text-mlb-muted text-center py-4">
                    No pitchers found
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Stats Panel */}
        <div className="lg:col-span-2">
          {selectedPitcher && stats ? (
            <PitcherStatsCard stats={stats} />
          ) : selectedPitcher && statsLoading ? (
            <div className="bg-mlb-card border border-mlb-border rounded-lg p-12 flex items-center justify-center">
              <Activity className="w-5 h-5 text-mlb-muted animate-spin" />
              <span className="ml-2 text-sm text-mlb-muted">Loading stats...</span>
            </div>
          ) : (
            <div className="bg-mlb-card border border-mlb-border rounded-lg p-12 text-center">
              <TrendingUp className="w-8 h-8 text-mlb-muted mx-auto mb-3" />
              <p className="text-sm text-mlb-muted">
                Select a pitcher to view their stats
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PitcherStatsCard({ stats }: { stats: PitcherStatsResult }) {
  const statRows = [
    { label: "ERA", value: stats.stats.era?.toFixed(2) ?? "—", desc: "Earned Run Average" },
    { label: "WHIP", value: stats.stats.whip?.toFixed(2) ?? "—", desc: "Walks + Hits per Inning" },
    { label: "K/9", value: stats.stats.k_per_9?.toFixed(1) ?? "—", desc: "Strikeouts per 9 innings" },
    { label: "BB/9", value: stats.stats.bb_per_9?.toFixed(1) ?? "—", desc: "Walks per 9 innings" },
  ];

  return (
    <div className="space-y-4">
      {/* Player Header */}
      <div className="bg-mlb-card border border-mlb-border rounded-lg p-5">
        <div className="flex items-center gap-4">
          <PlayerHeadshot mlbId={stats.mlb_id} name={stats.name} size="lg" />
          <div>
            <h2 className="text-base font-bold text-mlb-text">{stats.name}</h2>
            <p className="text-xs text-mlb-muted">
              {stats.team || "—"} &middot; {stats.throws ? `Throws ${stats.throws}` : "Pitcher"}
            </p>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {statRows.map((row) => (
          <div key={row.label} className="bg-mlb-card border border-mlb-border rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-mlb-text">{row.value}</p>
            <p className="text-xs font-semibold text-mlb-red mt-1">{row.label}</p>
            <p className="text-[10px] text-mlb-muted mt-0.5">{row.desc}</p>
          </div>
        ))}
      </div>

      {/* Volume Stats */}
      <div className="bg-mlb-card border border-mlb-border rounded-lg p-5">
        <h3 className="text-sm font-semibold text-mlb-text mb-3">Season Totals</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div>
            <p className="text-lg font-bold text-mlb-text">{stats.stats.total_games}</p>
            <p className="text-[10px] text-mlb-muted">Games</p>
          </div>
          <div>
            <p className="text-lg font-bold text-mlb-text">{stats.stats.total_innings.toFixed(1)}</p>
            <p className="text-[10px] text-mlb-muted">Innings</p>
          </div>
          <div>
            <p className="text-lg font-bold text-mlb-text">{stats.stats.total_strikeouts}</p>
            <p className="text-[10px] text-mlb-muted">Strikeouts</p>
          </div>
          <div>
            <p className="text-lg font-bold text-mlb-text">{stats.stats.total_walks}</p>
            <p className="text-[10px] text-mlb-muted">Walks</p>
          </div>
          <div>
            <p className="text-lg font-bold text-mlb-text">{stats.stats.total_earned_runs}</p>
            <p className="text-[10px] text-mlb-muted">Earned Runs</p>
          </div>
        </div>
      </div>
    </div>
  );
}
