"use client";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { getGamePredictions, getWinProbability } from "@/lib/api";
import type { GamePlayerPrediction, WinProbabilityResult } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import WinProbabilityChart from "@/components/charts/WinProbabilityChart";
import {
  Activity, MapPin, Clock, Users, AlertCircle, ChevronLeft, TrendingUp, BarChart3,
} from "lucide-react";
import Link from "next/link";

export default function GameDetailPage() {
  const params = useParams();
  const gameId = params.gameId as string;

  const { data, isLoading, error } = useQuery({
    queryKey: ["gamePredictions", gameId],
    queryFn: () => getGamePredictions(gameId),
    enabled: !!gameId,
  });

  const { data: winProb } = useQuery({
    queryKey: ["winProbability", gameId],
    queryFn: () => getWinProbability(gameId),
    enabled: !!gameId,
    retry: false,
  });

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-center py-20 gap-2">
          <div
            className="w-5 h-5 border-2 rounded-full animate-spin"
            style={{ borderColor: "var(--color-border)", borderTopColor: "var(--color-primary)" }}
          />
          <span className="text-sm" style={{ color: "var(--color-muted)" }}>Loading game data...</span>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-7xl mx-auto py-10 space-y-4">
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-1 text-xs transition-colors"
          style={{ color: "var(--color-muted)" }}
        >
          <ChevronLeft className="w-4 h-4" /> Back to Games
        </Link>
        <div
          className="rounded-xl p-8 text-center"
          style={{ background: "var(--color-card)", border: "1px solid rgba(249,115,22,0.3)" }}
        >
          <AlertCircle className="w-8 h-8 mx-auto mb-3" style={{ color: "#f97316" }} />
          <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--color-text)" }}>
            Unable to load game
          </h3>
          <p className="text-xs" style={{ color: "var(--color-muted)" }}>
            {error instanceof Error ? error.message : "Game data is not available."}
          </p>
        </div>
      </div>
    );
  }

  const { game, home_players, away_players } = data;
  const isLive = game.status?.toLowerCase().includes("progress");
  const isFinal = game.status?.toLowerCase().includes("final");

  return (
    <div className="max-w-7xl mx-auto space-y-5">
      <Link
        href="/dashboard"
        className="inline-flex items-center gap-1 text-xs transition-colors"
        style={{ color: "var(--color-muted)" }}
        onMouseEnter={e => (e.currentTarget.style.color = "var(--color-secondary)")}
        onMouseLeave={e => (e.currentTarget.style.color = "var(--color-muted)")}
      >
        <ChevronLeft className="w-4 h-4" /> Back to Games
      </Link>

      {/* Game Header */}
      <div
        className="rounded-xl p-6"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
      >
        <div className="flex items-center justify-between mb-4">
          <span
            className="text-[10px] font-bold uppercase px-2.5 py-1 rounded-full"
            style={{
              background: isLive ? "rgba(94,252,141,0.15)" : isFinal ? "rgba(147,190,223,0.1)" : "rgba(131,119,209,0.15)",
              color: isLive ? "var(--color-primary)" : isFinal ? "var(--color-muted)" : "var(--color-accent)",
            }}
          >
            {isLive && <span className="mr-1">●</span>}
            {game.status}
          </span>
          {game.venue && (
            <span className="flex items-center gap-1 text-[10px]" style={{ color: "var(--color-muted)" }}>
              <MapPin className="w-3 h-3" />
              {game.venue}
            </span>
          )}
        </div>

        <div className="grid grid-cols-3 items-center gap-4">
          {/* Away */}
          <div className="text-center">
            <p className="text-xl font-bold" style={{ color: "var(--color-text)" }}>{game.away_team}</p>
            {game.away_probable_pitcher && game.away_probable_pitcher !== "TBD" && (
              <p className="text-[10px] mt-1" style={{ color: "var(--color-muted)" }}>
                SP: {game.away_probable_pitcher}
              </p>
            )}
          </div>

          {/* Score / Time */}
          <div className="text-center">
            {game.away_score != null && game.home_score != null ? (
              <p className="text-4xl font-bold" style={{ color: "var(--color-text)" }}>
                {game.away_score} – {game.home_score}
              </p>
            ) : (
              <div className="flex items-center justify-center gap-1.5" style={{ color: "var(--color-muted)" }}>
                <Clock className="w-4 h-4" />
                <span className="text-base">
                  {game.game_datetime
                    ? new Date(game.game_datetime).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
                    : "TBD"}
                </span>
              </div>
            )}
            {game.game_date && (
              <p className="text-[10px] mt-1" style={{ color: "var(--color-subtle)" }}>{game.game_date}</p>
            )}
          </div>

          {/* Home */}
          <div className="text-center">
            <p className="text-xl font-bold" style={{ color: "var(--color-text)" }}>{game.home_team}</p>
            {game.home_probable_pitcher && game.home_probable_pitcher !== "TBD" && (
              <p className="text-[10px] mt-1" style={{ color: "var(--color-muted)" }}>
                SP: {game.home_probable_pitcher}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Statistical Analysis Card */}
      {winProb && <StatisticalAnalysisCard data={winProb} />}

      {/* Roster Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <TeamRosterPanel teamName={game.away_team} players={away_players} label="Away" />
        <TeamRosterPanel teamName={game.home_team} players={home_players} label="Home" />
      </div>

      {/* Legend */}
      <div
        className="rounded-xl p-4 text-xs"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)", color: "var(--color-muted)" }}
      >
        <p className="font-semibold uppercase tracking-wider mb-1.5" style={{ color: "var(--color-accent)" }}>
          About this data
        </p>
        <p>
          Player stats shown are <strong style={{ color: "var(--color-text)" }}>recent season statistics</strong> pulled from MLB Stats API and Statcast.
          Projected runs use Pythagorean expectation (wOBA × park factor × ERA adjustment) and are{" "}
          <strong style={{ color: "var(--color-text)" }}>statistical estimates</strong>, not guarantees.
          Win probability is based on the same methodology used by FanGraphs and Baseball Reference.
        </p>
      </div>
    </div>
  );
}

function StatisticalAnalysisCard({ data }: { data: WinProbabilityResult }) {
  const homeWinPct = Math.round(data.home_win_pct * 100);
  const awayWinPct = Math.round(data.away_win_pct * 100);
  const favored = homeWinPct > awayWinPct ? "home" : homeWinPct < awayWinPct ? "away" : "even";

  return (
    <div
      className="rounded-xl p-5"
      style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
    >
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp className="w-4 h-4" style={{ color: "var(--color-accent)" }} />
        <h3 className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>
          Statistical Analysis
        </h3>
        <span
          className="text-[10px] px-2 py-0.5 rounded ml-auto cursor-help"
          data-tooltip="Based on Pythagorean expectation using lineup wOBA, park factors, and starting pitcher ERA. Not a prediction — a probabilistic estimate."
          style={{ background: "rgba(131,119,209,0.15)", color: "var(--color-accent)" }}
        >
          Confidence: {Math.round(data.confidence * 100)}%
        </span>
      </div>

      {/* Win Prob Bar */}
      <div className="mb-4">
        <div className="flex justify-between mb-1 text-xs font-semibold">
          <span style={{ color: favored === "away" ? "var(--color-secondary)" : "var(--color-muted)" }}>
            {data.away.team_abbreviation} {awayWinPct}%
          </span>
          <span className="text-[10px]" style={{ color: "var(--color-subtle)" }}>Win Probability</span>
          <span style={{ color: favored === "home" ? "var(--color-primary)" : "var(--color-muted)" }}>
            {homeWinPct}% {data.home.team_abbreviation}
          </span>
        </div>
        <div className="win-prob-bar h-3 rounded-full">
          <div className="away-side" style={{ width: `${awayWinPct}%`, borderRadius: "3px 0 0 3px", transition: "width 0.5s" }} />
          <div className="home-side" style={{ width: `${homeWinPct}%`, borderRadius: "0 3px 3px 0", transition: "width 0.5s" }} />
        </div>
      </div>

      {/* Projected Runs */}
      <div className="grid grid-cols-2 gap-3">
        {[
          { team: data.away, color: "var(--color-secondary)" },
          { team: data.home, color: "var(--color-primary)" },
        ].map(({ team, color }) => (
          <div
            key={team.team_abbreviation}
            className="rounded-lg p-3"
            style={{ background: "var(--color-dark)", border: `1px solid ${color}30` }}
          >
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-3.5 h-3.5" style={{ color }} />
              <span className="text-xs font-semibold" style={{ color: "var(--color-text)" }}>
                {team.team_abbreviation}
              </span>
            </div>
            <div className="space-y-1">
              {[
                { label: "Projected Runs", value: team.projected_runs?.toFixed(1), highlight: true },
                { label: "Proj. Hits", value: team.projected_hits?.toFixed(1) },
                { label: "Proj. HR", value: team.projected_hr?.toFixed(1) },
              ].map(({ label, value, highlight }) => (
                <div key={label} className="flex justify-between text-[10px]">
                  <span style={{ color: "var(--color-muted)" }}>{label}</span>
                  <span
                    className="font-bold"
                    style={{ color: highlight ? color : "var(--color-text)" }}
                  >
                    {value ?? "—"}
                  </span>
                </div>
              ))}
              <div
                className="text-[9px] mt-1 pt-1"
                style={{ borderTop: "1px solid var(--color-border)", color: "var(--color-subtle)" }}
              >
                {team.n_players_with_predictions ?? 0} of {team.n_total_players ?? 0} players with data
              </div>
            </div>
          </div>
        ))}
      </div>

      <p className="text-[10px] mt-3 text-center" style={{ color: "var(--color-subtle)" }}>
        Based on Pythagorean expectation using weighted lineup wOBA, park factors, and starting pitcher ERA
      </p>
    </div>
  );
}

function TeamRosterPanel({
  teamName, players, label,
}: {
  teamName: string;
  players: GamePlayerPrediction[];
  label: string;
}) {
  const withData = players.filter(p => p.has_prediction);
  const withoutData = players.filter(p => !p.has_prediction);

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
      {/* Header */}
      <div
        className="px-4 py-3 flex items-center gap-2"
        style={{ background: "var(--color-panel)", borderBottom: "1px solid var(--color-border)" }}
      >
        <Users className="w-4 h-4" style={{ color: "var(--color-accent)" }} />
        <h3 className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>{teamName}</h3>
        <span className="ml-auto text-[10px]" style={{ color: "var(--color-muted)" }}>{label}</span>
      </div>

      {players.length === 0 ? (
        <div className="p-6 text-center" style={{ background: "var(--color-card)" }}>
          <AlertCircle className="w-5 h-5 mx-auto mb-2" style={{ color: "var(--color-muted)" }} />
          <p className="text-xs" style={{ color: "var(--color-muted)" }}>No roster data available for this team.</p>
        </div>
      ) : (
        <div style={{ background: "var(--color-card)" }}>
          {/* Column headers */}
          <div
            className="grid px-4 py-2 text-[10px] font-semibold uppercase tracking-wider"
            style={{
              gridTemplateColumns: "1fr 48px 48px 48px 48px",
              borderBottom: "1px solid var(--color-border)",
              color: "var(--color-muted)",
            }}
          >
            <span>Player</span>
            <span className="text-center" data-tooltip="Stat-based projected hits per game">H</span>
            <span className="text-center" data-tooltip="Stat-based projected home runs per game">HR</span>
            <span className="text-center" data-tooltip="Stat-based projected RBI per game">RBI</span>
            <span className="text-center" data-tooltip="Stat-based projected walks per game">BB</span>
          </div>

          {withData.map((p, idx) => (
            <PlayerRow key={p.player_id} player={p} idx={idx} />
          ))}

          {withoutData.length > 0 && (
            <>
              <div
                className="px-4 py-1.5 text-[10px]"
                style={{ background: "rgba(131,119,209,0.06)", color: "var(--color-subtle)", borderTop: "1px solid var(--color-border)" }}
              >
                {withoutData.length} player{withoutData.length !== 1 ? "s" : ""} without recent data
              </div>
              {withoutData.slice(0, 5).map((p, idx) => (
                <PlayerRow key={p.player_id} player={p} idx={withData.length + idx} />
              ))}
              {withoutData.length > 5 && (
                <div className="px-4 py-2 text-center text-[10px]" style={{ color: "var(--color-subtle)" }}>
                  +{withoutData.length - 5} more
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function PlayerRow({ player, idx }: { player: GamePlayerPrediction; idx: number }) {
  const even = idx % 2 === 0;
  return (
    <div
      className="grid items-center px-4 py-2 transition-colors"
      style={{
        gridTemplateColumns: "1fr 48px 48px 48px 48px",
        background: even ? "transparent" : "rgba(131,119,209,0.04)",
        borderBottom: "1px solid var(--color-border)",
      }}
      onMouseEnter={e => (e.currentTarget.style.background = "rgba(94,252,141,0.05)")}
      onMouseLeave={e => (e.currentTarget.style.background = even ? "transparent" : "rgba(131,119,209,0.04)")}
    >
      <div className="flex items-center gap-2 min-w-0">
        <PlayerHeadshot url={player.headshot_url} name={player.name} size="sm" />
        <div className="min-w-0">
          <p className="text-xs font-medium truncate" style={{ color: "var(--color-text)" }}>{player.name}</p>
          <p className="text-[10px]" style={{ color: "var(--color-muted)" }}>{player.position || "—"}</p>
        </div>
      </div>

      {player.has_prediction ? (
        <>
          <StatCell value={player.predicted_hits} color="var(--color-secondary)" />
          <StatCell value={player.predicted_hr} color="var(--color-primary)" bold />
          <StatCell value={player.predicted_rbi} />
          <StatCell value={player.predicted_walks} />
        </>
      ) : (
        <>
          {[0,1,2,3].map(i => (
            <span key={i} className="text-center text-[10px]" style={{ color: "var(--color-subtle)" }}>—</span>
          ))}
        </>
      )}
    </div>
  );
}

function StatCell({ value, color, bold }: { value?: number; color?: string; bold?: boolean }) {
  return (
    <p
      className="text-center text-xs tabular-nums"
      style={{
        color: color || "var(--color-text)",
        fontWeight: bold ? 700 : 500,
      }}
    >
      {value?.toFixed(2) ?? "—"}
    </p>
  );
}
