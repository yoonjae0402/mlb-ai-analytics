"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { searchPitchers, getPitcherStats } from "@/lib/api";
import type { Player, PitcherStatsResult } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import { TrendingUp, Search, AlertCircle } from "lucide-react";

const STAT_META = [
  {
    key: "era",
    label: "ERA",
    desc: "Earned Run Average",
    tip: "Earned runs allowed per 9 innings. Under 3.00 is ace territory; under 4.00 is solid.",
    goodBelow: 4.0,
    format: (v: number) => v.toFixed(2),
  },
  {
    key: "whip",
    label: "WHIP",
    desc: "Walks + Hits per Inning",
    tip: "How many runners per inning a pitcher allows. Under 1.20 is excellent; under 1.00 is elite.",
    goodBelow: 1.3,
    format: (v: number) => v.toFixed(2),
  },
  {
    key: "k_per_9",
    label: "K/9",
    desc: "Strikeouts per 9 Innings",
    tip: "Strikeout rate per 9 innings. Above 10 is elite for a starter.",
    goodAbove: 9.0,
    format: (v: number) => v.toFixed(1),
  },
  {
    key: "bb_per_9",
    label: "BB/9",
    desc: "Walks per 9 Innings",
    tip: "Walk rate per 9 innings. Under 3.0 is good control; under 2.0 is elite.",
    goodBelow: 3.5,
    format: (v: number) => v.toFixed(1),
  },
];

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

  const { data: stats, isLoading: statsLoading, error: statsError } = useQuery({
    queryKey: ["pitcherStats", selectedPitcher?.id],
    queryFn: () => getPitcherStats(selectedPitcher!.id),
    enabled: !!selectedPitcher,
  });

  const displayList = query.length > 0 ? pitchers : defaultPitchers;

  return (
    <div className="max-w-7xl mx-auto space-y-5">
      <div className="flex items-center gap-3">
        <TrendingUp className="w-5 h-5" style={{ color: "var(--color-primary)" }} />
        <div>
          <h1 className="text-lg font-bold" style={{ color: "var(--color-text)" }}>Pitcher Stats</h1>
          <p className="text-xs" style={{ color: "var(--color-muted)" }}>Search and view pitcher performance metrics</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Search Panel */}
        <div>
          <div
            className="rounded-xl overflow-hidden"
            style={{ border: "1px solid var(--color-border)" }}
          >
            {/* Search input */}
            <div
              className="p-3"
              style={{ background: "var(--color-panel)", borderBottom: "1px solid var(--color-border)" }}
            >
              <div
                className="flex items-center gap-2 px-3 py-2 rounded-lg"
                style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)" }}
              >
                <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--color-muted)" }} />
                <input
                  type="text"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder="Search pitchers..."
                  className="bg-transparent border-none outline-none text-sm flex-1"
                  style={{ color: "var(--color-text)" }}
                />
              </div>
            </div>

            {/* Pitcher list */}
            <div className="max-h-[480px] overflow-y-auto" style={{ background: "var(--color-card)" }}>
              {searchLoading ? (
                <div className="p-4 space-y-2">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <div className="skeleton w-8 h-8 rounded-full" />
                      <div className="flex-1">
                        <div className="skeleton h-3 w-24 rounded mb-1" />
                        <div className="skeleton h-2 w-16 rounded" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                (displayList || []).map((p) => (
                  <button
                    key={p.id}
                    onClick={() => setSelectedPitcher(p)}
                    className="w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors"
                    style={{
                      borderBottom: "1px solid var(--color-border)",
                      background: selectedPitcher?.id === p.id ? "rgba(94,252,141,0.08)" : "transparent",
                    }}
                    onMouseEnter={e => {
                      if (selectedPitcher?.id !== p.id)
                        (e.currentTarget as HTMLElement).style.background = "rgba(131,119,209,0.1)";
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLElement).style.background =
                        selectedPitcher?.id === p.id ? "rgba(94,252,141,0.08)" : "transparent";
                    }}
                  >
                    <PlayerHeadshot url={p.headshot_url} name={p.name} size="sm" />
                    <div className="min-w-0">
                      <p
                        className="text-xs font-medium truncate"
                        style={{ color: selectedPitcher?.id === p.id ? "var(--color-primary)" : "var(--color-text)" }}
                      >
                        {p.name}
                      </p>
                      <p className="text-[10px]" style={{ color: "var(--color-muted)" }}>
                        {p.team || "—"} · {p.throws ? `Throws ${p.throws}` : "P"}
                      </p>
                    </div>
                    {selectedPitcher?.id === p.id && (
                      <span className="ml-auto text-[9px] font-bold" style={{ color: "var(--color-primary)" }}>●</span>
                    )}
                  </button>
                ))
              )}
              {!searchLoading && displayList?.length === 0 && (
                <div className="p-6 text-center text-xs" style={{ color: "var(--color-muted)" }}>
                  No pitchers found
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Stats Panel */}
        <div className="lg:col-span-2">
          {!selectedPitcher && (
            <div
              className="rounded-xl p-12 text-center"
              style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
            >
              <TrendingUp className="w-8 h-8 mx-auto mb-3" style={{ color: "var(--color-muted)" }} />
              <p className="text-sm" style={{ color: "var(--color-muted)" }}>Select a pitcher to view stats</p>
            </div>
          )}

          {selectedPitcher && statsLoading && (
            <div
              className="rounded-xl p-12 flex items-center justify-center"
              style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
            >
              <div
                className="w-5 h-5 rounded-full border-2 animate-spin mr-2"
                style={{ borderColor: "var(--color-border)", borderTopColor: "var(--color-primary)" }}
              />
              <span className="text-sm" style={{ color: "var(--color-muted)" }}>Loading stats...</span>
            </div>
          )}

          {selectedPitcher && statsError && (
            <div
              className="rounded-xl p-6 text-center"
              style={{ background: "var(--color-card)", border: "1px solid rgba(249,115,22,0.3)" }}
            >
              <AlertCircle className="w-6 h-6 mx-auto mb-2" style={{ color: "#f97316" }} />
              <p className="text-sm" style={{ color: "#f97316" }}>Failed to load pitcher stats</p>
            </div>
          )}

          {selectedPitcher && stats && !statsLoading && (
            <PitcherStatsCard stats={stats} />
          )}
        </div>
      </div>
    </div>
  );
}

function PitcherStatsCard({ stats }: { stats: PitcherStatsResult }) {
  return (
    <div className="space-y-4">
      {/* Header */}
      <div
        className="rounded-xl p-5 flex items-center gap-4"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
      >
        <PlayerHeadshot url={stats.headshot_url} name={stats.name} size="lg" />
        <div>
          <h2 className="text-lg font-bold" style={{ color: "var(--color-text)" }}>{stats.name}</h2>
          <p className="text-xs" style={{ color: "var(--color-muted)" }}>
            {stats.team || "—"} · {stats.throws ? `Throws ${stats.throws}` : "Pitcher"}
          </p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {STAT_META.map((meta) => {
          const val = stats.stats[meta.key as keyof typeof stats.stats] as number | null;
          const isGood = val != null && (
            meta.goodBelow != null ? val <= meta.goodBelow : val >= (meta.goodAbove ?? 0)
          );

          return (
            <div
              key={meta.key}
              className="rounded-xl p-4 text-center"
              style={{
                background: "var(--color-card)",
                border: `1px solid ${isGood ? "rgba(94,252,141,0.3)" : "var(--color-border)"}`,
              }}
              data-tooltip={meta.tip}
            >
              <p
                className="text-2xl font-bold"
                style={{ color: isGood ? "var(--color-primary)" : "var(--color-text)" }}
              >
                {val != null ? meta.format(val) : "—"}
              </p>
              <p
                className="text-xs font-semibold mt-1 cursor-help"
                style={{
                  color: "var(--color-accent)",
                  borderBottom: "1px dotted var(--color-accent)",
                }}
              >
                {meta.label}
              </p>
              <p className="text-[10px] mt-0.5" style={{ color: "var(--color-muted)" }}>
                {meta.desc}
              </p>
              {isGood && (
                <span
                  className="mt-1 inline-block text-[9px] font-bold px-1.5 py-0.5 rounded"
                  style={{ background: "rgba(94,252,141,0.15)", color: "var(--color-primary)" }}
                >
                  GOOD
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Season Totals */}
      <div
        className="rounded-xl p-5"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
      >
        <div
          className="text-[10px] font-semibold uppercase tracking-wider mb-3"
          style={{ color: "var(--color-muted)" }}
        >
          Season Totals
        </div>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { label: "Games", value: stats.stats.total_games },
            { label: "Innings", value: stats.stats.total_innings?.toFixed(1) },
            { label: "Strikeouts", value: stats.stats.total_strikeouts },
            { label: "Walks", value: stats.stats.total_walks },
            { label: "Earned Runs", value: stats.stats.total_earned_runs },
          ].map(({ label, value }) => (
            <div key={label}>
              <p className="text-lg font-bold" style={{ color: "var(--color-text)" }}>{value ?? "—"}</p>
              <p className="text-[10px]" style={{ color: "var(--color-muted)" }}>{label}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
