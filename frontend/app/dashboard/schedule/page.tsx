"use client";
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { getScheduleRange } from "@/lib/api";
import type { ScheduleGame } from "@/lib/api";
import {
  Calendar, ChevronLeft, ChevronRight, Clock, MapPin, ExternalLink,
} from "lucide-react";

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

  // Compute range based on view
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
      // Pad to full weeks
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

  // Group games by date
  const gamesByDate = useMemo(() => {
    const map: Record<string, ScheduleGame[]> = {};
    for (const g of games) {
      const d = g.game_date;
      if (!map[d]) map[d] = [];
      map[d].push(g);
    }
    return map;
  }, [games]);

  const navigateBack = () => {
    if (view === "week") {
      setCurrentDate(addDays(currentDate, -7));
    } else {
      setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
    }
  };

  const navigateForward = () => {
    if (view === "week") {
      setCurrentDate(addDays(currentDate, 7));
    } else {
      setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));
    }
  };

  const today = formatDate(new Date());

  return (
    <div className="max-w-7xl mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Calendar className="w-5 h-5 text-mlb-red" />
          <div>
            <h1 className="text-lg font-bold text-mlb-text">Schedule</h1>
            <p className="text-xs text-mlb-muted">
              {currentDate.toLocaleDateString("en-US", {
                month: "long",
                year: "numeric",
              })}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* View Toggle */}
          <div className="flex bg-mlb-surface border border-mlb-border rounded overflow-hidden">
            <button
              onClick={() => setView("week")}
              className={`px-3 py-1 text-xs ${
                view === "week"
                  ? "bg-mlb-red text-white"
                  : "text-mlb-muted hover:text-mlb-text"
              }`}
            >
              Week
            </button>
            <button
              onClick={() => setView("month")}
              className={`px-3 py-1 text-xs ${
                view === "month"
                  ? "bg-mlb-red text-white"
                  : "text-mlb-muted hover:text-mlb-text"
              }`}
            >
              Month
            </button>
          </div>

          {/* Navigation */}
          <button
            onClick={() => setCurrentDate(new Date())}
            className="px-3 py-1 text-xs bg-mlb-surface border border-mlb-border rounded text-mlb-muted hover:text-mlb-text"
          >
            Today
          </button>
          <button onClick={navigateBack} className="p-1 rounded bg-mlb-surface border border-mlb-border text-mlb-muted hover:text-mlb-text">
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button onClick={navigateForward} className="p-1 rounded bg-mlb-surface border border-mlb-border text-mlb-muted hover:text-mlb-text">
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Calendar Grid */}
      <div className="bg-mlb-card border border-mlb-border rounded-lg overflow-hidden">
        {/* Day Headers */}
        <div className="grid grid-cols-7 border-b border-mlb-border">
          {WEEKDAYS.map((day) => (
            <div key={day} className="px-2 py-2 text-center text-[10px] font-semibold text-mlb-muted uppercase">
              {day}
            </div>
          ))}
        </div>

        {/* Day Cells */}
        <div className="grid grid-cols-7">
          {days.map((day, i) => {
            const key = formatDate(day);
            const dayGames = gamesByDate[key] || [];
            const isToday = key === today;
            const isCurrentMonth = day.getMonth() === currentDate.getMonth();

            return (
              <div
                key={i}
                className={`border-b border-r border-mlb-border/50 p-1.5 ${
                  view === "week" ? "min-h-[200px]" : "min-h-[80px]"
                } ${!isCurrentMonth && view === "month" ? "opacity-40" : ""}`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span
                    className={`text-[11px] font-medium px-1.5 py-0.5 rounded ${
                      isToday
                        ? "bg-mlb-red text-white"
                        : "text-mlb-muted"
                    }`}
                  >
                    {day.getDate()}
                  </span>
                  {dayGames.length > 0 && (
                    <span className="text-[9px] text-mlb-muted">
                      {dayGames.length}g
                    </span>
                  )}
                </div>

                <div className="space-y-0.5">
                  {dayGames.slice(0, view === "week" ? 10 : 3).map((g) => (
                    <ScheduleGameCard key={g.game_id} game={g} compact={view === "month"} />
                  ))}
                  {view === "month" && dayGames.length > 3 && (
                    <p className="text-[9px] text-mlb-muted text-center">
                      +{dayGames.length - 3} more
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {isLoading && (
        <div className="text-center py-4">
          <p className="text-xs text-mlb-muted">Loading schedule...</p>
        </div>
      )}
    </div>
  );
}

function ScheduleGameCard({ game, compact }: { game: ScheduleGame; compact: boolean }) {
  const gameUrl = `/dashboard/game/${game.game_id}`;

  if (compact) {
    return (
      <Link
        href={gameUrl}
        className="block text-[9px] text-mlb-text bg-mlb-surface/50 rounded px-1 py-0.5 truncate hover:bg-mlb-blue/10 hover:text-mlb-blue transition-colors"
      >
        {game.away_team} @ {game.home_team}
      </Link>
    );
  }

  const isFinal = game.status?.toLowerCase().includes("final");
  const isLive = game.status?.toLowerCase().includes("in progress");

  return (
    <Link
      href={gameUrl}
      className="block bg-mlb-surface/50 rounded p-1.5 text-[10px] hover:bg-mlb-blue/10 hover:border-mlb-blue/30 transition-colors group"
    >
      <div className="flex items-center justify-between">
        <span className="font-medium text-mlb-text truncate group-hover:text-mlb-blue">
          {game.away_team} @ {game.home_team}
        </span>
        {isLive ? (
          <span className="text-mlb-red font-semibold text-[8px]">LIVE</span>
        ) : (
          <ExternalLink className="w-2.5 h-2.5 text-mlb-muted opacity-0 group-hover:opacity-100 transition-opacity" />
        )}
      </div>
      {(isFinal || isLive) && game.away_score != null && (
        <div className="text-mlb-muted mt-0.5">
          {game.away_score} - {game.home_score}
          {isFinal && " F"}
        </div>
      )}
      {!isFinal && !isLive && (
        <div className="flex items-center gap-1 text-mlb-muted mt-0.5">
          <Clock className="w-2.5 h-2.5" />
          <span>{game.game_datetime ? new Date(game.game_datetime).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : game.status}</span>
        </div>
      )}
      {game.home_probable_pitcher && game.home_probable_pitcher !== "TBD" && (
        <div className="text-mlb-muted mt-0.5 truncate">
          {game.away_probable_pitcher || "TBD"} vs {game.home_probable_pitcher}
        </div>
      )}
    </Link>
  );
}
