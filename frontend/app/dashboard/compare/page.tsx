"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { usePlayerSearch } from "@/hooks/usePlayerSearch";
import { comparePlayers } from "@/lib/api";
import type { Player, PlayerDetail } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import ContextBadge from "@/components/ui/ContextBadge";
import PercentileBar from "@/components/ui/PercentileBar";
import TrendIndicator from "@/components/ui/TrendIndicator";
import StatTooltip from "@/components/ui/StatTooltip";
import {
  Search, Scale, X, AlertCircle,
} from "lucide-react";
import {
  getContextLevel,
  STAT_DISPLAY,
  formatStatValue,
} from "@/lib/stat-helpers";

const COMPARE_STATS = [
  { key: "batting_avg", seasonKey: "batting_avg" },
  { key: "obp", seasonKey: "obp" },
  { key: "slg", seasonKey: "slg" },
  { key: "hits", seasonKey: "hits" },
  { key: "home_runs", seasonKey: "home_runs" },
  { key: "rbi", seasonKey: "rbi" },
  { key: "walks", seasonKey: "walks" },
];

export default function ComparePage() {
  const [selectedPlayers, setSelectedPlayers] = useState<Player[]>([]);
  const [activeSearch, setActiveSearch] = useState<0 | 1>(0);

  const search1 = usePlayerSearch();
  const search2 = usePlayerSearch();

  const playerIds = selectedPlayers.map((p) => p.id);
  const canCompare = playerIds.length === 2;

  const { data: compareData, isLoading: isComparing } = useQuery({
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
        <Scale className="w-5 h-5 text-mlb-blue" />
        <div>
          <h1 className="text-lg font-bold text-mlb-text">Compare Players</h1>
          <p className="text-xs text-mlb-muted">Side-by-side stat comparison with context</p>
        </div>
      </div>

      {/* Player Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PlayerSelector
          label="Player 1"
          player={selectedPlayers[0]}
          search={search1}
          onSelect={(p) => handleSelectPlayer(p, 0)}
          onRemove={() => handleRemovePlayer(0)}
        />
        <PlayerSelector
          label="Player 2"
          player={selectedPlayers[1]}
          search={search2}
          onSelect={(p) => handleSelectPlayer(p, 1)}
          onRemove={() => handleRemovePlayer(1)}
        />
      </div>

      {/* Comparison Results */}
      {canCompare && isComparing && (
        <div className="text-center py-8">
          <div className="w-6 h-6 border-2 border-mlb-blue border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-xs text-mlb-muted mt-2">Loading comparison...</p>
        </div>
      )}

      {canCompare && compareData && compareData.players.length === 2 && (
        <ComparisonTable
          player1={compareData.players[0]}
          player2={compareData.players[1]}
        />
      )}

      {!canCompare && (
        <div className="bg-mlb-card border border-mlb-border rounded-lg p-8 text-center">
          <Scale className="w-8 h-8 text-mlb-muted mx-auto mb-3" />
          <p className="text-sm text-mlb-muted">Select two players to compare their stats</p>
        </div>
      )}
    </div>
  );
}

function PlayerSelector({
  label,
  player,
  search,
  onSelect,
  onRemove,
}: {
  label: string;
  player?: Player;
  search: ReturnType<typeof usePlayerSearch>;
  onSelect: (p: Player) => void;
  onRemove: () => void;
}) {
  if (player) {
    return (
      <div className="bg-mlb-card border border-mlb-blue/30 rounded-xl p-4 flex items-center gap-3">
        <PlayerHeadshot url={player.headshot_url} name={player.name} size="md" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-mlb-text truncate">{player.name}</p>
          <p className="text-[10px] text-mlb-muted">{player.team} - {player.position}</p>
        </div>
        <button
          onClick={onRemove}
          className="p-1 rounded-md hover:bg-mlb-surface text-mlb-muted hover:text-mlb-red transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <p className="text-[10px] font-semibold text-mlb-muted uppercase mb-2">{label}</p>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-mlb-muted" />
        <input
          type="text"
          value={search.query}
          onChange={(e) => search.setQuery(e.target.value)}
          placeholder="Search players by name..."
          className="w-full bg-mlb-surface border border-mlb-border rounded-lg pl-9 pr-3 py-2 text-sm text-mlb-text placeholder-mlb-muted focus:border-mlb-blue"
        />
      </div>

      {search.data && search.data.length > 0 && (
        <div className="mt-2 max-h-48 overflow-y-auto space-y-1">
          {search.data.map((p: Player) => (
            <button
              key={p.id}
              onClick={() => onSelect(p)}
              className="w-full flex items-center gap-2 p-2 rounded-md hover:bg-mlb-surface/50 transition-colors text-left"
            >
              <PlayerHeadshot url={p.headshot_url} name={p.name} size="sm" />
              <div className="min-w-0">
                <p className="text-xs font-medium text-mlb-text truncate">{p.name}</p>
                <p className="text-[10px] text-mlb-muted">{p.team} - {p.position}</p>
              </div>
            </button>
          ))}
        </div>
      )}

      {search.query.length >= 2 && search.data && search.data.length === 0 && (
        <p className="text-[10px] text-mlb-muted mt-2 text-center">No players found</p>
      )}
    </div>
  );
}

function ComparisonTable({
  player1,
  player2,
}: {
  player1: PlayerDetail;
  player2: PlayerDetail;
}) {
  const stats1 = player1.season_totals || {};
  const stats2 = player2.season_totals || {};

  // Compute recent averages from recent_stats for trend
  function getRecentAvg(pd: PlayerDetail, key: string): number {
    const recent = pd.recent_stats.slice(0, 10);
    if (recent.length === 0) return 0;
    const vals = recent.map((s: any) => s[key] ?? 0);
    return vals.reduce((a: number, b: number) => a + b, 0) / vals.length;
  }

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl overflow-hidden">
      {/* Header */}
      <div className="grid grid-cols-[1fr_1fr_1fr] border-b border-mlb-border">
        <div className="p-4 text-center border-r border-mlb-border">
          <PlayerHeadshot url={player1.player.headshot_url} name={player1.player.name} size="md" />
          <p className="text-sm font-semibold text-mlb-text mt-2">{player1.player.name}</p>
          <p className="text-[10px] text-mlb-muted">{player1.player.team}</p>
        </div>
        <div className="p-4 flex items-center justify-center">
          <span className="text-xs font-semibold text-mlb-muted uppercase">Stat</span>
        </div>
        <div className="p-4 text-center border-l border-mlb-border">
          <PlayerHeadshot url={player2.player.headshot_url} name={player2.player.name} size="md" />
          <p className="text-sm font-semibold text-mlb-text mt-2">{player2.player.name}</p>
          <p className="text-[10px] text-mlb-muted">{player2.player.team}</p>
        </div>
      </div>

      {/* Stat Rows */}
      {COMPARE_STATS.map(({ key }) => {
        const val1 = stats1[key] ?? 0;
        const val2 = stats2[key] ?? 0;
        const level1 = getContextLevel(key, val1);
        const level2 = getContextLevel(key, val2);

        // Who wins?
        const winner = val1 > val2 ? 1 : val2 > val1 ? 2 : 0;

        // Trend
        const recentAvg1 = getRecentAvg(player1, key);
        const recentAvg2 = getRecentAvg(player2, key);

        return (
          <div
            key={key}
            className="grid grid-cols-[1fr_1fr_1fr] border-b border-mlb-border/50 hover:bg-mlb-surface/20"
          >
            {/* Player 1 value */}
            <div className={`p-3 flex items-center justify-between border-r border-mlb-border/50 ${
              winner === 1 ? "bg-mlb-green/5" : ""
            }`}>
              <div className="space-y-1">
                <p className={`text-sm font-bold tabular-nums ${
                  winner === 1 ? "text-mlb-green" : "text-mlb-text"
                }`}>
                  {formatStatValue(key, val1)}
                </p>
                <PercentileBar stat={key} value={val1} />
              </div>
              <div className="flex items-center gap-1.5">
                <TrendIndicator recent={recentAvg1} season={val1} />
                <ContextBadge stat={key} value={val1} />
              </div>
            </div>

            {/* Stat Name */}
            <div className="p-3 flex items-center justify-center">
              <span className="text-xs font-semibold text-mlb-muted">
                {STAT_DISPLAY[key] || key}
              </span>
              <StatTooltip stat={key} />
            </div>

            {/* Player 2 value */}
            <div className={`p-3 flex items-center justify-between border-l border-mlb-border/50 ${
              winner === 2 ? "bg-mlb-green/5" : ""
            }`}>
              <div className="flex items-center gap-1.5">
                <ContextBadge stat={key} value={val2} />
                <TrendIndicator recent={recentAvg2} season={val2} />
              </div>
              <div className="space-y-1 text-right">
                <p className={`text-sm font-bold tabular-nums ${
                  winner === 2 ? "text-mlb-green" : "text-mlb-text"
                }`}>
                  {formatStatValue(key, val2)}
                </p>
                <PercentileBar stat={key} value={val2} />
              </div>
            </div>
          </div>
        );
      })}

      {/* Games played */}
      <div className="grid grid-cols-[1fr_1fr_1fr] bg-mlb-surface/30">
        <div className="p-3 text-center border-r border-mlb-border/50">
          <span className="text-xs text-mlb-muted">{stats1.games ?? 0} games</span>
        </div>
        <div className="p-3 text-center">
          <span className="text-[10px] font-semibold text-mlb-muted uppercase">Games</span>
        </div>
        <div className="p-3 text-center border-l border-mlb-border/50">
          <span className="text-xs text-mlb-muted">{stats2.games ?? 0} games</span>
        </div>
      </div>
    </div>
  );
}
