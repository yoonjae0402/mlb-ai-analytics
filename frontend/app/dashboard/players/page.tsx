"use client";
import { useState } from "react";
import Link from "next/link";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { getPlayerIndex, getTeams } from "@/lib/api";
import type { Player } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import {
  Users, Search, ChevronLeft, ChevronRight, Filter,
} from "lucide-react";

const LEVELS = ["MLB", "AAA", "AA", "A+", "A"];
const POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "P"];

export default function PlayersPage() {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [teamFilter, setTeamFilter] = useState("");
  const [levelFilter, setLevelFilter] = useState("");
  const [positionFilter, setPositionFilter] = useState("");

  const { data: teamsData } = useQuery({
    queryKey: ["teams"],
    queryFn: getTeams,
  });

  const { data, isLoading } = useQuery({
    queryKey: ["playerIndex", page, search, teamFilter, levelFilter, positionFilter],
    queryFn: () =>
      getPlayerIndex({
        page,
        per_page: 25,
        search: search || undefined,
        team: teamFilter || undefined,
        level: levelFilter || undefined,
        position: positionFilter || undefined,
      }),
    placeholderData: keepPreviousData,
  });

  const players = data?.players || [];
  const total = data?.total || 0;
  const totalPages = Math.ceil(total / 25);

  return (
    <div className="max-w-7xl mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Users className="w-5 h-5 text-mlb-red" />
        <div>
          <h1 className="text-lg font-bold text-mlb-text">Player Index</h1>
          <p className="text-xs text-mlb-muted">
            {total.toLocaleString()} players — MLB + MiLB
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-mlb-card border border-mlb-border rounded-lg p-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 flex-1 min-w-[200px]">
            <Search className="w-4 h-4 text-mlb-muted" />
            <input
              type="text"
              placeholder="Search by name..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              className="flex-1 bg-mlb-surface border border-mlb-border rounded px-3 py-1.5 text-sm text-mlb-text placeholder-mlb-muted"
            />
          </div>

          <div className="flex items-center gap-2">
            <Filter className="w-3.5 h-3.5 text-mlb-muted" />

            <select
              value={teamFilter}
              onChange={(e) => {
                setTeamFilter(e.target.value);
                setPage(1);
              }}
              className="bg-mlb-surface border border-mlb-border rounded px-2 py-1.5 text-xs text-mlb-text"
            >
              <option value="">All Teams</option>
              {teamsData?.map((t) => (
                <option key={t.id} value={t.abbreviation}>
                  {t.abbreviation}
                </option>
              ))}
            </select>

            <select
              value={levelFilter}
              onChange={(e) => {
                setLevelFilter(e.target.value);
                setPage(1);
              }}
              className="bg-mlb-surface border border-mlb-border rounded px-2 py-1.5 text-xs text-mlb-text"
            >
              <option value="">All Levels</option>
              {LEVELS.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>

            <select
              value={positionFilter}
              onChange={(e) => {
                setPositionFilter(e.target.value);
                setPage(1);
              }}
              className="bg-mlb-surface border border-mlb-border rounded px-2 py-1.5 text-xs text-mlb-text"
            >
              <option value="">All Positions</option>
              {POSITIONS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-mlb-card border border-mlb-border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-mlb-border text-[11px] font-semibold text-mlb-muted uppercase tracking-wider">
              <th className="text-left px-4 py-2.5">Player</th>
              <th className="text-left px-3 py-2.5">Team</th>
              <th className="text-left px-3 py-2.5">Pos</th>
              <th className="text-left px-3 py-2.5">Level</th>
              <th className="text-left px-3 py-2.5">Bats/Throws</th>
              <th className="text-right px-4 py-2.5">Prospect Rank</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={6} className="text-center py-8 text-xs text-mlb-muted">
                  Loading players...
                </td>
              </tr>
            ) : players.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-8 text-xs text-mlb-muted">
                  No players found
                </td>
              </tr>
            ) : (
              players.map((player: Player) => (
                <tr
                  key={player.id}
                  className="border-b border-mlb-border/50 hover:bg-mlb-surface/50 transition-colors"
                >
                  <td className="px-4 py-2">
                    <Link
                      href={`/dashboard/player/${player.id}`}
                      className="flex items-center gap-2.5 group"
                    >
                      <PlayerHeadshot
                        url={player.headshot_url}
                        name={player.name}
                        size="sm"
                      />
                      <span className="text-sm font-medium text-mlb-text group-hover:text-mlb-blue transition-colors">
                        {player.name}
                      </span>
                    </Link>
                  </td>
                  <td className="px-3 py-2 text-xs text-mlb-muted">
                    {player.team || "—"}
                  </td>
                  <td className="px-3 py-2 text-xs text-mlb-muted">
                    {player.position || "—"}
                  </td>
                  <td className="px-3 py-2">
                    <LevelBadge level={player.current_level} />
                  </td>
                  <td className="px-3 py-2 text-xs text-mlb-muted">
                    {player.bats || "—"}/{player.throws || "—"}
                  </td>
                  <td className="px-4 py-2 text-right">
                    {player.prospect_rank ? (
                      <span className="text-xs font-semibold text-mlb-red">
                        #{player.prospect_rank}
                      </span>
                    ) : (
                      <span className="text-xs text-mlb-muted">—</span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-xs text-mlb-muted">
            Page {page} of {totalPages}
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="p-1.5 rounded bg-mlb-surface border border-mlb-border text-mlb-muted hover:text-mlb-text disabled:opacity-30"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPage(Math.min(totalPages, page + 1))}
              disabled={page === totalPages}
              className="p-1.5 rounded bg-mlb-surface border border-mlb-border text-mlb-muted hover:text-mlb-text disabled:opacity-30"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function LevelBadge({ level }: { level?: string }) {
  const colors: Record<string, string> = {
    MLB: "bg-mlb-red/15 text-mlb-red",
    AAA: "bg-blue-500/15 text-blue-400",
    AA: "bg-green-500/15 text-green-400",
    "A+": "bg-yellow-500/15 text-yellow-400",
    A: "bg-purple-500/15 text-purple-400",
  };

  const cls = colors[level || ""] || "bg-mlb-surface text-mlb-muted";
  return (
    <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${cls}`}>
      {level || "—"}
    </span>
  );
}
