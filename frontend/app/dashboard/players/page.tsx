"use client";
import { useState } from "react";
import Link from "next/link";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { getPlayerIndex, getTeams } from "@/lib/api";
import type { Player } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import { Users, Search, ChevronLeft, ChevronRight, Filter } from "lucide-react";

const LEVELS = ["MLB", "AAA", "AA", "A+", "A"];
const POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "P"];

const LEVEL_COLORS: Record<string, { bg: string; color: string }> = {
  MLB: { bg: "rgba(94,252,141,0.15)", color: "var(--color-primary)" },
  AAA: { bg: "rgba(142,249,243,0.15)", color: "var(--color-secondary)" },
  AA:  { bg: "rgba(147,190,223,0.15)", color: "var(--color-accent)" },
  "A+":{ bg: "rgba(245,158,11,0.15)", color: "#f59e0b" },
  A:   { bg: "rgba(131,119,209,0.2)", color: "var(--color-panel)" },
};

export default function PlayersPage() {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [teamFilter, setTeamFilter] = useState("");
  const [levelFilter, setLevelFilter] = useState("");
  const [positionFilter, setPositionFilter] = useState("");

  const { data: teamsData } = useQuery({ queryKey: ["teams"], queryFn: getTeams });

  const { data, isLoading } = useQuery({
    queryKey: ["playerIndex", page, search, teamFilter, levelFilter, positionFilter],
    queryFn: () =>
      getPlayerIndex({
        page, per_page: 25,
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
    <div className="max-w-7xl mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Users className="w-5 h-5" style={{ color: "var(--color-primary)" }} />
        <div>
          <h1 className="text-lg font-bold" style={{ color: "var(--color-text)" }}>Player Index</h1>
          <p className="text-xs" style={{ color: "var(--color-muted)" }}>
            {total.toLocaleString()} players — MLB + MiLB
          </p>
        </div>
      </div>

      {/* Filters bar */}
      <div
        className="rounded-xl p-3"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
      >
        <div className="flex flex-wrap items-center gap-3">
          {/* Search */}
          <div
            className="flex items-center gap-2 flex-1 min-w-[200px] px-3 py-2 rounded-lg"
            style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)" }}
          >
            <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--color-muted)" }} />
            <input
              type="text"
              placeholder="Search by name..."
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(1); }}
              className="flex-1 bg-transparent border-none outline-none text-sm"
              style={{ color: "var(--color-text)" }}
            />
          </div>

          {/* Filters */}
          <div className="flex items-center gap-2">
            <Filter className="w-3.5 h-3.5" style={{ color: "var(--color-muted)" }} />
            <select
              value={teamFilter}
              onChange={e => { setTeamFilter(e.target.value); setPage(1); }}
              style={{ fontSize: "12px", padding: "6px 28px 6px 8px", minWidth: "80px" }}
            >
              <option value="">All Teams</option>
              {teamsData?.map(t => (
                <option key={t.id} value={t.abbreviation}>{t.abbreviation}</option>
              ))}
            </select>
            <select
              value={levelFilter}
              onChange={e => { setLevelFilter(e.target.value); setPage(1); }}
              style={{ fontSize: "12px", padding: "6px 28px 6px 8px", minWidth: "80px" }}
            >
              <option value="">All Levels</option>
              {LEVELS.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
            <select
              value={positionFilter}
              onChange={e => { setPositionFilter(e.target.value); setPage(1); }}
              style={{ fontSize: "12px", padding: "6px 28px 6px 8px", minWidth: "70px" }}
            >
              <option value="">All Pos</option>
              {POSITIONS.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
        <table className="fg-table w-full">
          <thead>
            <tr>
              <th>Player</th>
              <th>
                <abbr
                  className="stat-abbr"
                  data-tip="Team abbreviation (MLB franchise or minor league affiliate)"
                >
                  Team
                </abbr>
              </th>
              <th>
                <abbr className="stat-abbr" data-tip="Fielding position (1B = first base, SS = shortstop, etc.)">Pos</abbr>
              </th>
              <th>Level</th>
              <th>
                <abbr className="stat-abbr" data-tip="Bats (R=Right, L=Left, S=Switch) / Throws (R/L)">B/T</abbr>
              </th>
              <th style={{ textAlign: "right" }}>
                <abbr className="stat-abbr" data-tip="Baseball America / MLB Pipeline prospect ranking">Rank</abbr>
              </th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={i}>
                  <td>
                    <div className="flex items-center gap-2">
                      <div className="skeleton w-7 h-7 rounded-full" />
                      <div className="skeleton h-3 w-28 rounded" />
                    </div>
                  </td>
                  {Array.from({ length: 5 }).map((_, j) => (
                    <td key={j}><div className="skeleton h-3 w-12 rounded" /></td>
                  ))}
                </tr>
              ))
            ) : players.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: "center", padding: "32px", color: "var(--color-muted)" }}>
                  No players found matching your filters.
                </td>
              </tr>
            ) : (
              players.map((player: Player) => (
                <tr key={player.id}>
                  <td>
                    <Link
                      href={`/dashboard/player/${player.id}`}
                      className="flex items-center gap-2.5 group"
                      style={{ textDecoration: "none" }}
                    >
                      <PlayerHeadshot url={player.headshot_url} name={player.name} size="sm" />
                      <span
                        className="font-medium text-sm group-hover:underline"
                        style={{ color: "var(--color-secondary)" }}
                      >
                        {player.name}
                      </span>
                    </Link>
                  </td>
                  <td style={{ color: "var(--color-muted)", fontSize: "12px" }}>
                    {player.team || "—"}
                  </td>
                  <td style={{ color: "var(--color-muted)", fontSize: "12px" }}>
                    {player.position || "—"}
                  </td>
                  <td>
                    <LevelBadge level={player.current_level} />
                  </td>
                  <td style={{ color: "var(--color-muted)", fontSize: "12px" }}>
                    {player.bats || "—"}/{player.throws || "—"}
                  </td>
                  <td className="numeric">
                    {player.prospect_rank ? (
                      <span className="font-bold" style={{ color: "var(--color-primary)" }}>
                        #{player.prospect_rank}
                      </span>
                    ) : (
                      <span style={{ color: "var(--color-subtle)" }}>—</span>
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
          <p className="text-xs" style={{ color: "var(--color-muted)" }}>
            Page {page} of {totalPages} ({total.toLocaleString()} total players)
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="p-1.5 rounded transition-colors disabled:opacity-30"
              style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)", color: "var(--color-muted)" }}
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            {/* Page numbers */}
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              const start = Math.max(1, Math.min(page - 2, totalPages - 4));
              const p = start + i;
              return (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  className="w-7 h-7 rounded text-xs font-medium transition-colors"
                  style={{
                    background: p === page ? "var(--color-panel)" : "var(--color-dark)",
                    color: p === page ? "var(--color-primary)" : "var(--color-muted)",
                    border: `1px solid ${p === page ? "var(--color-primary)" : "var(--color-border)"}`,
                  }}
                >
                  {p}
                </button>
              );
            })}
            <button
              onClick={() => setPage(Math.min(totalPages, page + 1))}
              disabled={page === totalPages}
              className="p-1.5 rounded transition-colors disabled:opacity-30"
              style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)", color: "var(--color-muted)" }}
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
  const style = LEVEL_COLORS[level || ""] || { bg: "rgba(147,190,223,0.1)", color: "var(--color-muted)" };
  return (
    <span
      className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
      style={{ background: style.bg, color: style.color }}
    >
      {level || "—"}
    </span>
  );
}
