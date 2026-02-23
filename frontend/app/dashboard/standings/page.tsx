"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getStandings } from "@/lib/api";
import type { DivisionStandings, TeamStanding } from "@/lib/api";
import Link from "next/link";
import { Trophy, AlertCircle, TrendingUp } from "lucide-react";

const DIVISIONS = [
  "AL East", "AL Central", "AL West",
  "NL East", "NL Central", "NL West",
];

const COL_TIPS: Record<string, string> = {
  W:    "Wins — games the team has won this season",
  L:    "Losses — games the team has lost this season",
  PCT:  "Win percentage — wins divided by total games played",
  GB:   "Games Behind — how many games behind the division leader this team is",
  Streak: "Current win (W) or loss (L) streak",
  L10:  "Record over the last 10 games (e.g. 7-3 means 7 wins, 3 losses)",
  Home: "Home record — wins-losses at their home stadium",
  Away: "Away record — wins-losses on the road",
};

export default function StandingsPage() {
  const [activeTab, setActiveTab] = useState("AL East");

  const { data, isLoading, error } = useQuery({
    queryKey: ["standings"],
    queryFn: getStandings,
    refetchInterval: 600_000, // 10 minutes
    staleTime: 300_000,       // 5 minutes
  });

  const activeDiv = data?.divisions.find(d => d.division_name === activeTab);

  return (
    <div className="max-w-5xl mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Trophy className="w-5 h-5" style={{ color: "#f59e0b" }} />
        <div>
          <h1 className="text-xl font-bold" style={{ color: "var(--color-text)" }}>
            MLB Standings
          </h1>
          <p className="text-xs" style={{ color: "var(--color-muted)" }}>
            {data ? `As of ${data.as_of}` : "Current season standings by division"}
            {" · "}
            <span className="beginner-label">Top 3 teams make the playoffs. Wild cards give extra spots!</span>
            <span className="advanced-stat">Top 3 per division + 3 wild cards per league make the postseason.</span>
          </p>
        </div>
      </div>

      {/* Division Tabs */}
      <div
        className="flex overflow-x-auto gap-1 p-1 rounded-xl"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
      >
        {DIVISIONS.map(div => {
          const isActive = activeTab === div;
          const league = div.startsWith("AL") ? "AL" : "NL";
          return (
            <button
              key={div}
              onClick={() => setActiveTab(div)}
              className="flex-shrink-0 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
              style={{
                background: isActive ? "var(--color-panel)" : "transparent",
                color: isActive ? "var(--color-primary)" : "var(--color-muted)",
                border: isActive ? "1px solid var(--color-primary)" : "1px solid transparent",
              }}
            >
              <span
                className="text-[9px] font-bold mr-1"
                style={{ color: league === "AL" ? "var(--color-secondary)" : "var(--color-accent)" }}
              >
                {league}
              </span>
              {div.replace(/^(AL|NL)\s/, "")}
            </button>
          );
        })}
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="flex items-center gap-4 px-4 py-3"
              style={{ borderBottom: "1px solid var(--color-border)" }}
            >
              <div className="skeleton w-5 h-4 rounded" />
              <div className="skeleton h-3 w-40 rounded" />
              <div className="ml-auto flex gap-6">
                {Array.from({ length: 6 }).map((_, j) => (
                  <div key={j} className="skeleton h-3 w-8 rounded" />
                ))}
              </div>
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
          <p className="text-sm" style={{ color: "#f97316" }}>Failed to load standings.</p>
          <p className="text-xs mt-1" style={{ color: "var(--color-muted)" }}>
            Make sure the backend is running and has network access to the MLB Stats API.
          </p>
        </div>
      )}

      {/* Standings Table */}
      {!isLoading && !error && activeDiv && (
        <StandingsTable division={activeDiv} />
      )}

      {/* Beginner explanation */}
      <div
        className="rounded-xl p-4 text-xs beginner-label"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)", color: "var(--color-muted)" }}
      >
        <strong style={{ color: "var(--color-accent)" }}>How playoffs work:</strong>{" "}
        Each league (AL and NL) sends 6 teams to the playoffs — the top team from each of the 3 divisions,
        plus 3 wild card teams (the next-best records regardless of division).
        The <span style={{ color: "var(--color-primary)" }}>division winners</span> get a first-round bye.
      </div>

      <div
        className="rounded-xl p-4 text-xs advanced-stat"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)", color: "var(--color-muted)" }}
      >
        <strong style={{ color: "var(--color-accent)" }}>Postseason format:</strong>{" "}
        3 division winners + 3 wild cards per league. Wild card teams are ranked by record across
        all non-division-winner teams. GB shows games behind the division leader.
        Green rows = division leaders. Yellow rows = wild card contenders.
      </div>
    </div>
  );
}

function StandingsTable({ division }: { division: DivisionStandings }) {
  return (
    <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
      {/* Column headers */}
      <div
        className="grid px-4 py-2.5 text-[10px] font-semibold uppercase tracking-wider"
        style={{
          gridTemplateColumns: "24px 1fr 48px 48px 64px 52px 52px 56px 56px",
          background: "var(--color-panel)",
          borderBottom: "2px solid var(--color-primary)",
          color: "var(--color-muted)",
        }}
      >
        <span>#</span>
        <span>Team</span>
        <span className="text-right">
          <abbr className="stat-abbr" data-tip={COL_TIPS.W}>W</abbr>
        </span>
        <span className="text-right">
          <abbr className="stat-abbr" data-tip={COL_TIPS.L}>L</abbr>
        </span>
        <span className="text-right">
          <span className="advanced-stat">
            <abbr className="stat-abbr" data-tip={COL_TIPS.PCT}>PCT</abbr>
          </span>
          <span className="beginner-label">
            <abbr className="stat-abbr" data-tip={COL_TIPS.PCT}>Win%</abbr>
          </span>
        </span>
        <span className="text-right">
          <abbr className="stat-abbr" data-tip={COL_TIPS.GB}>GB</abbr>
        </span>
        <span className="text-right">
          <abbr className="stat-abbr" data-tip={COL_TIPS.Streak}>Strk</abbr>
        </span>
        <span className="text-right">
          <abbr className="stat-abbr" data-tip={COL_TIPS.L10}>L10</abbr>
        </span>
        <span className="text-right hide-mobile">
          <abbr className="stat-abbr" data-tip={COL_TIPS.Home}>Home</abbr>
        </span>
      </div>

      {/* Team rows */}
      {division.teams.map((team, idx) => (
        <StandingsRow key={team.team_name} team={team} idx={idx} />
      ))}

      {/* Legend */}
      <div
        className="flex items-center gap-4 px-4 py-2 text-[10px]"
        style={{ background: "var(--color-deeper)", borderTop: "1px solid var(--color-border)", color: "var(--color-subtle)" }}
      >
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-sm" style={{ background: "rgba(94,252,141,0.15)" }} />
          <span>Division leader</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-sm" style={{ background: "rgba(245,158,11,0.1)" }} />
          <span>Wild card contender</span>
        </div>
      </div>
    </div>
  );
}

function StandingsRow({ team, idx }: { team: TeamStanding; idx: number }) {
  const isLeader = team.division_rank === 1;
  const isWildcard = team.division_rank >= 2 && team.division_rank <= 4; // rough WC indicator

  const bg = isLeader
    ? "rgba(94,252,141,0.08)"
    : isWildcard
    ? "rgba(245,158,11,0.05)"
    : idx % 2 === 0
    ? "transparent"
    : "rgba(131,119,209,0.06)";

  const streakIsWin = team.streak.startsWith("W");
  const streakColor = streakIsWin ? "var(--color-primary)" : "#f97316";

  return (
    <div
      className="grid items-center px-4 py-2.5 transition-colors"
      style={{
        gridTemplateColumns: "24px 1fr 48px 48px 64px 52px 52px 56px 56px",
        borderBottom: "1px solid var(--color-border)",
        background: bg,
      }}
      onMouseEnter={e => (e.currentTarget.style.background = "rgba(94,252,141,0.06)")}
      onMouseLeave={e => (e.currentTarget.style.background = bg)}
    >
      {/* Rank */}
      <span
        className="text-xs font-bold"
        style={{ color: isLeader ? "var(--color-primary)" : "var(--color-subtle)" }}
      >
        {team.division_rank}
      </span>

      {/* Team name */}
      <Link
        href={`/dashboard/players?team=${team.team_abbreviation}`}
        className="flex items-center gap-1.5 group"
        style={{ textDecoration: "none" }}
      >
        <span
          className="text-[11px] font-bold group-hover:underline"
          style={{ color: isLeader ? "var(--color-primary)" : "var(--color-text)" }}
        >
          {team.team_name}
        </span>
        {isLeader && (
          <TrendingUp className="w-3 h-3" style={{ color: "var(--color-primary)" }} />
        )}
      </Link>

      {/* W */}
      <span className="text-right text-xs tabular-nums font-semibold" style={{ color: "var(--color-text)" }}>
        {team.wins}
      </span>

      {/* L */}
      <span className="text-right text-xs tabular-nums" style={{ color: "var(--color-muted)" }}>
        {team.losses}
      </span>

      {/* PCT */}
      <span className="text-right text-xs tabular-nums font-mono" style={{ color: "var(--color-secondary)" }}>
        {team.pct.toFixed(3)}
      </span>

      {/* GB */}
      <span className="text-right text-xs tabular-nums" style={{ color: "var(--color-muted)" }}>
        {team.gb}
      </span>

      {/* Streak */}
      <span className="text-right text-xs tabular-nums font-semibold" style={{ color: streakColor }}>
        {team.streak}
      </span>

      {/* L10 */}
      <span className="text-right text-xs tabular-nums" style={{ color: "var(--color-muted)" }}>
        {team.last_10}
      </span>

      {/* Home */}
      <span className="text-right text-xs tabular-nums hide-mobile" style={{ color: "var(--color-subtle)" }}>
        {team.home_record || "—"}
      </span>
    </div>
  );
}
