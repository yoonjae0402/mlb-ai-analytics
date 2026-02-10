"use client";
import { usePlayerSearch } from "@/hooks/usePlayerSearch";
import PlayerCard from "@/components/cards/PlayerCard";
import type { Player } from "@/lib/api";
import { Search } from "lucide-react";

interface PlayerSearchProps {
  onSelect: (player: Player) => void;
  selectedId?: number;
}

export default function PlayerSearch({ onSelect, selectedId }: PlayerSearchProps) {
  const { query, setQuery, data: players, isLoading, error } = usePlayerSearch();

  return (
    <div>
      <div className="relative mb-3">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-mlb-muted" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search players by name..."
          className="w-full bg-mlb-surface border border-mlb-border rounded-lg pl-10 pr-4 py-2 text-sm text-mlb-text placeholder:text-mlb-muted focus:outline-none focus:border-mlb-blue"
        />
      </div>

      {isLoading && (
        <p className="text-xs text-mlb-muted">Searching...</p>
      )}

      {error && (
        <p className="text-xs text-mlb-red">
          Search unavailable — make sure the backend is running.
        </p>
      )}

      {players && players.length > 0 && (
        <div className="space-y-2 max-h-60 overflow-y-auto">
          {players.map((player) => (
            <PlayerCard
              key={player.id}
              player={player}
              selected={player.id === selectedId}
              onClick={() => onSelect(player)}
            />
          ))}
        </div>
      )}

      {players && players.length === 0 && query.length >= 2 && !error && (
        <p className="text-xs text-mlb-muted">
          No players found for &ldquo;{query}&rdquo;. Try a different name.
        </p>
      )}
    </div>
  );
}
