"use client";
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { getScheduleRange } from "@/lib/api";
import type { ScheduleGame } from "@/lib/api";
import { Calendar, ChevronLeft, ChevronRight, Clock, ExternalLink } from "lucide-react";

function formatDate(d: Date): string {
  return d.toISOString().split("T")[0];
}
function addDays(d: Date, n: number): Date {
  const result = new Date(d);
  result.setDate(result.getDate() + n);
  return result;
}
function startOfWeek(d: Date): Date {
  const result = new Date(d);
  result.setDate(result.getDate() - result.getDay());
  return result;
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

export default function SchedulePage() {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [view, setView] = useState<"week" | "month">("week");

  const { startDate, endDate, days } = useMemo(() => {
    if (view === "week") {
      const start = startOfWeek(currentDate);
      const end = addDays(start, 6);
      const days: Date[] = [];
      for (let i = 0; i < 7; i++) days.push(addDays(start, i));
      return { startDate: formatDate(start), endDate: formatDate(end), days };
    } else {
      const start = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1);
      const end = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0);
      const calStart = startOfWeek(start);
      const days: Date[] = [];
      let d = calStart;
      while (d <= end || days.length % 7 !== 0) {
        days.push(new Date(d));
        d = addDays(d, 1);
        if (days.length > 42) break;
      }
      return { startDate: formatDate(start), endDate: formatDate(end), days };
    }
  }, [currentDate, view]);

  const { data, isLoading } = useQuery({
    queryKey: ["schedule", startDate, endDate],
    queryFn: () => getScheduleRange(startDate, endDate),
  });

  const games = data?.games || [];

  const gamesByDate = useMemo(() => {
    const map: Record<string, ScheduleGame[]> = {};
    for (const g of games) {
      const d = g.game_date;
      if (!map[d]) map[d] = [];
      map[d].push(g);
    }
    return map;
  }, [games]);

  const today = formatDate(new Date());

  const navigateBack = () => {
    if (view === "week") setCurrentDate(addDays(currentDate, -7));
    else setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
  };
  const navigateForward = () => {
    if (view === "week") setCurrentDate(addDays(currentDate, 7));
    else setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));
  };

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Calendar className="w-5 h-5" style={{ color: "var(--color-primary)" }} />
          <div>
            <h1 className="text-lg font-bold" style={{ color: "var(--color-text)" }}>
              Schedule
            </h1>
            <p className="text-xs" style={{ color: "var(--color-muted)" }}>
              {currentDate.toLocaleDateString("en-US", { month: "long", year: "numeric" })}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* View toggle */}
          <div
            className="flex rounded overflow-hidden"
            style={{ border: "1px solid var(--color-border)", background: "var(--color-dark)" }}
          >
            {(["week", "month"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className="px-3 py-1.5 text-xs font-medium capitalize transition-colors"
                style={{
                  background: view === v ? "var(--color-panel)" : "transparent",
                  color: view === v ? "var(--color-primary)" : "var(--color-muted)",
                }}
              >
                {v}
              </button>
            ))}
          </div>

          <button
            onClick={() => setCurrentDate(new Date())}
            className="px-3 py-1.5 text-xs rounded"
            style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)", color: "var(--color-muted)" }}
          >
            Today
          </button>
          <button
            onClick={navigateBack}
            className="p-1.5 rounded"
            style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)", color: "var(--color-muted)" }}
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={navigateForward}
            className="p-1.5 rounded"
            style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)", color: "var(--color-muted)" }}
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="text-center py-2">
          <div className="inline-flex items-center gap-2 text-xs" style={{ color: "var(--color-muted)" }}>
            <div className="w-3 h-3 border-2 rounded-full animate-spin"
              style={{ borderColor: "var(--color-border)", borderTopColor: "var(--color-primary)" }} />
            Loading schedule...
          </div>
        </div>
      )}

      {/* Calendar Grid */}
      <div
        className="rounded-xl overflow-hidden"
        style={{ border: "1px solid var(--color-border)" }}
      >
        {/* Day headers */}
        <div
          className="grid grid-cols-7"
          style={{ background: "var(--color-panel)", borderBottom: "2px solid var(--color-primary)" }}
        >
          {WEEKDAYS.map((day) => (
            <div
              key={day}
              className="px-2 py-2 text-center text-[10px] font-semibold uppercase tracking-wider"
              style={{ color: "var(--color-muted)" }}
            >
              {day}
            </div>
          ))}
        </div>

        {/* Day cells */}
        <div className="grid grid-cols-7" style={{ background: "var(--color-card)" }}>
          {days.map((day, i) => {
            const key = formatDate(day);
            const dayGames = gamesByDate[key] || [];
            const isToday = key === today;
            const isCurrentMonth = day.getMonth() === currentDate.getMonth();

            return (
              <div
                key={i}
                className={`p-1.5 ${view === "week" ? "min-h-[180px]" : "min-h-[80px]"}`}
                style={{
                  borderRight: "1px solid var(--color-border)",
                  borderBottom: "1px solid var(--color-border)",
                  opacity: !isCurrentMonth && view === "month" ? 0.35 : 1,
                  background: isToday ? "rgba(94,252,141,0.04)" : "transparent",
                }}
              >
                {/* Day number */}
                <div className="flex items-center justify-between mb-1">
                  <span
                    className="text-[11px] font-medium px-1.5 py-0.5 rounded-full"
                    style={{
                      background: isToday ? "var(--color-primary)" : "transparent",
                      color: isToday ? "#1a1a2e" : "var(--color-muted)",
                      fontWeight: isToday ? 700 : 400,
                    }}
                  >
                    {day.getDate()}
                  </span>
                  {dayGames.length > 0 && (
                    <span className="text-[9px]" style={{ color: "var(--color-subtle)" }}>
                      {dayGames.length}g
                    </span>
                  )}
                </div>

                {/* Games */}
                <div className="space-y-0.5">
                  {dayGames.slice(0, view === "week" ? 12 : 3).map((g) => (
                    <ScheduleGameCard key={g.game_id} game={g} compact={view === "month"} />
                  ))}
                  {view === "month" && dayGames.length > 3 && (
                    <p className="text-[9px] text-center" style={{ color: "var(--color-subtle)" }}>
                      +{dayGames.length - 3} more
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Empty state */}
      {!isLoading && games.length === 0 && (
        <div className="text-center py-6" style={{ color: "var(--color-muted)" }}>
          <p className="text-sm">No games scheduled for this period.</p>
        </div>
      )}
    </div>
  );
}

function ScheduleGameCard({ game, compact }: { game: ScheduleGame; compact: boolean }) {
  const gameUrl = `/dashboard/game/${game.game_id}`;
  const isFinal = game.status?.toLowerCase().includes("final");
  const isLive = game.status?.toLowerCase().includes("in progress") || game.status?.toLowerCase() === "live";

  if (compact) {
    return (
      <Link
        href={gameUrl}
        className="block text-[9px] px-1.5 py-0.5 rounded truncate transition-colors"
        style={{
          background: isLive ? "rgba(94,252,141,0.1)" : "rgba(131,119,209,0.1)",
          color: isLive ? "var(--color-primary)" : "var(--color-text)",
        }}
      >
        {game.away_team} @ {game.home_team}
        {game.home_win_prob != null && (
          <span style={{ color: "var(--color-muted)" }}> {Math.round(game.home_win_prob * 100)}%</span>
        )}
      </Link>
    );
  }

  return (
    <Link
      href={gameUrl}
      className="block rounded p-1.5 text-[10px] transition-all group"
      style={{
        background: isLive ? "rgba(94,252,141,0.06)" : "rgba(131,119,209,0.06)",
        border: `1px solid ${isLive ? "rgba(94,252,141,0.2)" : "var(--color-border)"}`,
        textDecoration: "none",
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.borderColor = "var(--color-secondary)";
        (e.currentTarget as HTMLElement).style.background = "rgba(142,249,243,0.08)";
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.borderColor = isLive ? "rgba(94,252,141,0.2)" : "var(--color-border)";
        (e.currentTarget as HTMLElement).style.background = isLive ? "rgba(94,252,141,0.06)" : "rgba(131,119,209,0.06)";
      }}
    >
      {/* Teams */}
      <div className="flex items-center justify-between mb-0.5">
        <span className="font-semibold truncate" style={{ color: "var(--color-text)" }}>
          {game.away_team} @ {game.home_team}
        </span>
        {isLive ? (
          <span className="text-[8px] font-bold flex items-center gap-0.5" style={{ color: "var(--color-primary)" }}>
            <span className="w-1 h-1 rounded-full live-dot" style={{ background: "var(--color-primary)" }} />
            LIVE
          </span>
        ) : (
          <ExternalLink className="w-2.5 h-2.5 opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: "var(--color-muted)" }} />
        )}
      </div>

      {/* Score or time */}
      {(isFinal || isLive) && game.away_score != null ? (
        <div style={{ color: "var(--color-muted)" }}>
          {game.away_score} – {game.home_score}
          {isFinal && <span style={{ color: "var(--color-accent)" }}> F</span>}
        </div>
      ) : (
        <div className="flex items-center gap-1" style={{ color: "var(--color-muted)" }}>
          <Clock className="w-2.5 h-2.5" />
          <span>
            {game.game_datetime
              ? new Date(game.game_datetime).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
              : game.status}
          </span>
        </div>
      )}

      {/* Pitchers */}
      {(game.away_probable_pitcher || game.home_probable_pitcher) && (
        <div className="mt-0.5 truncate" style={{ color: "var(--color-subtle)", fontSize: "9px" }}>
          {game.away_probable_pitcher || "TBD"} vs {game.home_probable_pitcher || "TBD"}
        </div>
      )}

      {/* Win Probability bar */}
      {game.home_win_prob != null && (
        <div className="mt-1.5">
          <div className="flex justify-between text-[8px] mb-0.5" style={{ color: "var(--color-subtle)" }}>
            <span>{Math.round((1 - game.home_win_prob) * 100)}%</span>
            <span>{Math.round(game.home_win_prob * 100)}%</span>
          </div>
          <div className="win-prob-bar">
            <div className="away-side" style={{ width: `${(1 - game.home_win_prob) * 100}%` }} />
            <div className="home-side" style={{ width: `${game.home_win_prob * 100}%` }} />
          </div>
        </div>
      )}
    </Link>
  );
}
