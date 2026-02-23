"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { usePlayerSearch } from "@/hooks/usePlayerSearch";
import { comparePlayers } from "@/lib/api";
import type { Player, PlayerDetail } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import { Search, Scale, X, AlertCircle } from "lucide-react";
import { getContextLevel, STAT_DISPLAY, formatStatValue } from "@/lib/stat-helpers";

const COMPARE_STATS = [
  { key: "batting_avg", tip: "Hits divided by at-bats" },
  { key: "obp", tip: "How often a batter reaches base (hits + walks + HBP)" },
  { key: "slg", tip: "Total bases divided by at-bats — measures power" },
  { key: "hits", tip: "Total hits" },
  { key: "home_runs", tip: "Home runs" },
  { key: "rbi", tip: "Runs batted in" },
  { key: "walks", tip: "Base on balls (walks)" },
];

const CONTEXT_COLORS: Record<string, { bg: string; color: string; label: string }> = {
  elite: { bg: "rgba(245,158,11,0.15)", color: "#f59e0b", label: "Elite" },
  great: { bg: "rgba(94,252,141,0.15)", color: "var(--color-primary)", label: "Great" },
  average: { bg: "rgba(147,190,223,0.1)", color: "var(--color-accent)", label: "Average" },
  below_avg: { bg: "rgba(249,115,22,0.1)", color: "#f97316", label: "Below Avg" },
};

export default function ComparePage() {
  const [selectedPlayers, setSelectedPlayers] = useState<Player[]>([]);
  const search1 = usePlayerSearch();
  const search2 = usePlayerSearch();

  const playerIds = selectedPlayers.map(p => p.id);
  const canCompare = playerIds.length === 2;

  const { data: compareData, isLoading: isComparing, error: compareError } = useQuery({
    queryKey: ["compare", ...playerIds],
    queryFn: () => comparePlayers(playerIds),
    enabled: canCompare,
  });

  const handleSelectPlayer = (player: Player, slot: 0 | 1) => {
    const newPlayers = [...selectedPlayers];
    newPlayers[slot] = player;
    setSelectedPlayers(newPlayers);
    if (slot === 0) search1.setQuery("");
    else search2.setQuery("");
  };

  const handleRemovePlayer = (slot: 0 | 1) => {
    const newPlayers = [...selectedPlayers];
    newPlayers.splice(slot, 1);
    setSelectedPlayers(newPlayers);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Scale className="w-5 h-5" style={{ color: "var(--color-secondary)" }} />
        <div>
          <h1 className="text-lg font-bold" style={{ color: "var(--color-text)" }}>Compare Players</h1>
          <p className="text-xs" style={{ color: "var(--color-muted)" }}>
            Side-by-side stat comparison with context badges and beginner-friendly explanations
          </p>
        </div>
      </div>

      {/* Player Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {([0, 1] as const).map(slot => (
          <PlayerSelector
            key={slot}
            label={`Player ${slot + 1}`}
            player={selectedPlayers[slot]}
            search={slot === 0 ? search1 : search2}
            onSelect={p => handleSelectPlayer(p, slot)}
            onRemove={() => handleRemovePlayer(slot)}
          />
        ))}
      </div>

      {/* Loading */}
      {canCompare && isComparing && (
        <div className="text-center py-8 flex items-center justify-center gap-2">
          <div
            className="w-5 h-5 border-2 rounded-full animate-spin"
            style={{ borderColor: "var(--color-border)", borderTopColor: "var(--color-secondary)" }}
          />
          <span className="text-sm" style={{ color: "var(--color-muted)" }}>Loading comparison...</span>
        </div>
      )}

      {/* Error */}
      {canCompare && compareError && (
        <div
          className="rounded-xl p-6 text-center"
          style={{ background: "var(--color-card)", border: "1px solid rgba(249,115,22,0.3)" }}
        >
          <AlertCircle className="w-5 h-5 mx-auto mb-2" style={{ color: "#f97316" }} />
          <p className="text-sm" style={{ color: "#f97316" }}>Failed to load comparison data.</p>
        </div>
      )}

      {/* Comparison Table */}
      {canCompare && compareData && compareData.players.length === 2 && (
        <ComparisonTable player1={compareData.players[0]} player2={compareData.players[1]} />
      )}

      {/* Empty state */}
      {!canCompare && (
        <div
          className="rounded-xl p-10 text-center"
          style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
        >
          <Scale className="w-8 h-8 mx-auto mb-3" style={{ color: "var(--color-muted)" }} />
          <p className="text-sm" style={{ color: "var(--color-muted)" }}>Select two players above to compare their stats</p>
          <p className="text-xs mt-1" style={{ color: "var(--color-subtle)" }}>
            Each stat shows a beginner context badge (Elite / Great / Average / Below Avg) and who leads
          </p>
        </div>
      )}
    </div>
  );
}

function PlayerSelector({
  label, player, search, onSelect, onRemove,
}: {
  label: string;
  player?: Player;
  search: ReturnType<typeof usePlayerSearch>;
  onSelect: (p: Player) => void;
  onRemove: () => void;
}) {
  if (player) {
    return (
      <div
        className="rounded-xl p-4 flex items-center gap-3"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-secondary)" }}
      >
        <PlayerHeadshot url={player.headshot_url} name={player.name} size="md" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold truncate" style={{ color: "var(--color-text)" }}>
            {player.name}
          </p>
          <p className="text-[10px]" style={{ color: "var(--color-muted)" }}>
            {player.team} · {player.position}
          </p>
        </div>
        <button
          onClick={onRemove}
          className="p-1 rounded transition-colors"
          style={{ color: "var(--color-muted)" }}
          onMouseEnter={e => (e.currentTarget.style.color = "#f97316")}
          onMouseLeave={e => (e.currentTarget.style.color = "var(--color-muted)")}
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div
      className="rounded-xl p-4"
      style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
    >
      <p className="text-[10px] font-semibold uppercase mb-2" style={{ color: "var(--color-muted)" }}>
        {label}
      </p>
      <div
        className="flex items-center gap-2 px-3 py-2 rounded-lg mb-2"
        style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)" }}
      >
        <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--color-muted)" }} />
        <input
          type="text"
          value={search.query}
          onChange={e => search.setQuery(e.target.value)}
          placeholder="Search players by name..."
          className="bg-transparent border-none outline-none text-sm flex-1"
          style={{ color: "var(--color-text)" }}
        />
      </div>

      {search.data && search.data.length > 0 && (
        <div className="max-h-48 overflow-y-auto rounded-lg" style={{ border: "1px solid var(--color-border)" }}>
          {search.data.map((p: Player) => (
            <button
              key={p.id}
              onClick={() => onSelect(p)}
              className="w-full flex items-center gap-2 p-2.5 text-left transition-colors"
              style={{ borderBottom: "1px solid var(--color-border)" }}
              onMouseEnter={e => (e.currentTarget.style.background = "rgba(94,252,141,0.07)")}
              onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
            >
              <PlayerHeadshot url={p.headshot_url} name={p.name} size="sm" />
              <div className="min-w-0">
                <p className="text-xs font-medium truncate" style={{ color: "var(--color-text)" }}>{p.name}</p>
                <p className="text-[10px]" style={{ color: "var(--color-muted)" }}>{p.team} · {p.position}</p>
              </div>
            </button>
          ))}
        </div>
      )}

      {search.query.length >= 2 && search.data?.length === 0 && (
        <p className="text-[10px] text-center mt-2" style={{ color: "var(--color-muted)" }}>No players found</p>
      )}
    </div>
  );
}

function ComparisonTable({ player1, player2 }: { player1: PlayerDetail; player2: PlayerDetail }) {
  const stats1 = player1.season_totals || {};
  const stats2 = player2.season_totals || {};

  function getRecentAvg(pd: PlayerDetail, key: string): number {
    const recent = pd.recent_stats.slice(0, 10);
    if (recent.length === 0) return 0;
    const vals = recent.map((s) => ((s as unknown) as Record<string, number>)[key] ?? 0);
    return vals.reduce((a: number, b: number) => a + b, 0) / vals.length;
  }

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
      {/* Player headers */}
      <div
        className="grid grid-cols-[1fr_1fr_1fr]"
        style={{ background: "var(--color-panel)", borderBottom: "2px solid var(--color-primary)" }}
      >
        <div className="p-4 text-center" style={{ borderRight: "1px solid var(--color-border)" }}>
          <PlayerHeadshot url={player1.player.headshot_url} name={player1.player.name} size="md" />
          <p className="text-sm font-semibold mt-2" style={{ color: "var(--color-text)" }}>{player1.player.name}</p>
          <p className="text-[10px]" style={{ color: "var(--color-muted)" }}>{player1.player.team}</p>
        </div>
        <div className="p-4 flex items-center justify-center">
          <span className="text-xs font-bold uppercase" style={{ color: "var(--color-muted)" }}>vs</span>
        </div>
        <div className="p-4 text-center" style={{ borderLeft: "1px solid var(--color-border)" }}>
          <PlayerHeadshot url={player2.player.headshot_url} name={player2.player.name} size="md" />
          <p className="text-sm font-semibold mt-2" style={{ color: "var(--color-text)" }}>{player2.player.name}</p>
          <p className="text-[10px]" style={{ color: "var(--color-muted)" }}>{player2.player.team}</p>
        </div>
      </div>

      {/* Stat rows */}
      {COMPARE_STATS.map(({ key, tip }, idx) => {
        const val1 = stats1[key] ?? 0;
        const val2 = stats2[key] ?? 0;
        const level1 = getContextLevel(key, val1);
        const level2 = getContextLevel(key, val2);
        const winner = val1 > val2 ? 1 : val2 > val1 ? 2 : 0;
        const even = idx % 2 === 0;

        const ctx1 = CONTEXT_COLORS[level1] || { bg: "transparent", color: "var(--color-muted)", label: level1 };
        const ctx2 = CONTEXT_COLORS[level2] || { bg: "transparent", color: "var(--color-muted)", label: level2 };

        return (
          <div
            key={key}
            className="grid grid-cols-[1fr_1fr_1fr]"
            style={{
              borderBottom: "1px solid var(--color-border)",
              background: even ? "transparent" : "rgba(131,119,209,0.04)",
            }}
          >
            {/* P1 value */}
            <div
              className="p-3 flex items-center justify-between"
              style={{
                borderRight: "1px solid var(--color-border)",
                background: winner === 1 ? "rgba(94,252,141,0.06)" : "transparent",
              }}
            >
              <div>
                <p
                  className="text-sm font-bold tabular-nums"
                  style={{ color: winner === 1 ? "var(--color-primary)" : "var(--color-text)" }}
                >
                  {formatStatValue(key, val1)}
                </p>
              </div>
              <span
                className="text-[9px] font-semibold px-1.5 py-0.5 rounded"
                style={{ background: ctx1.bg, color: ctx1.color }}
              >
                {ctx1.label}
              </span>
            </div>

            {/* Stat name */}
            <div className="p-3 flex items-center justify-center text-center">
              <div>
                <span
                  className="text-xs font-semibold cursor-help"
                  data-tooltip={tip}
                  style={{ color: "var(--color-accent)", borderBottom: "1px dotted var(--color-accent)" }}
                >
                  {STAT_DISPLAY[key] || key}
                </span>
                {winner !== 0 && (
                  <div className="text-[9px] mt-0.5" style={{ color: "var(--color-primary)" }}>
                    P{winner} leads
                  </div>
                )}
              </div>
            </div>

            {/* P2 value */}
            <div
              className="p-3 flex items-center justify-between"
              style={{
                borderLeft: "1px solid var(--color-border)",
                background: winner === 2 ? "rgba(94,252,141,0.06)" : "transparent",
              }}
            >
              <span
                className="text-[9px] font-semibold px-1.5 py-0.5 rounded"
                style={{ background: ctx2.bg, color: ctx2.color }}
              >
                {ctx2.label}
              </span>
              <p
                className="text-sm font-bold tabular-nums text-right"
                style={{ color: winner === 2 ? "var(--color-primary)" : "var(--color-text)" }}
              >
                {formatStatValue(key, val2)}
              </p>
            </div>
          </div>
        );
      })}

      {/* Games footer */}
      <div
        className="grid grid-cols-[1fr_1fr_1fr] text-xs"
        style={{ background: "var(--color-panel)", color: "var(--color-muted)" }}
      >
        <div className="p-3 text-center" style={{ borderRight: "1px solid var(--color-border)" }}>
          {stats1.games ?? 0} games
        </div>
        <div className="p-3 flex items-center justify-center">
          <span className="text-[10px] font-semibold uppercase">Season Games</span>
        </div>
        <div className="p-3 text-center" style={{ borderLeft: "1px solid var(--color-border)" }}>
          {stats2.games ?? 0} games
        </div>
      </div>
    </div>
  );
}
