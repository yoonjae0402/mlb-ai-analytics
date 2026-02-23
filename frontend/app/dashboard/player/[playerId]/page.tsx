"use client";
import { useParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { getPlayer, type PlayerStat } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import { formatStatValue, getContextLevel, BEGINNER_NAMES, getTrend } from "@/lib/stat-helpers";
import ContextBadge from "@/components/ui/ContextBadge";
import PercentileBar from "@/components/ui/PercentileBar";
import { ArrowLeft, BarChart2, History, User, Zap } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

// Core season stats with display metadata
const CORE_STAT_META: Record<string, { label: string; tip: string; format?: (v: number) => string }> = {
  batting_avg: { label: "AVG",  tip: "Batting average — hits divided by at-bats", format: v => v.toFixed(3) },
  obp:         { label: "OBP",  tip: "On-base percentage — how often batter reaches base", format: v => v.toFixed(3) },
  slg:         { label: "SLG",  tip: "Slugging — total bases per at-bat, measures power", format: v => v.toFixed(3) },
  hits:        { label: "H",    tip: "Total hits this season" },
  home_runs:   { label: "HR",   tip: "Home runs this season" },
  rbi:         { label: "RBI",  tip: "Runs batted in this season" },
  walks:       { label: "BB",   tip: "Walks (base on balls) this season" },
  games:       { label: "G",    tip: "Games played this season" },
};

// Advanced stats (hidden in beginner mode)
const ADVANCED_STAT_META: Record<string, { label: string; tip: string }> = {
  iso:   { label: "ISO",   tip: "Isolated Power = SLG - AVG. Measures raw power excluding singles. .150+ is good." },
  babip: { label: "BABIP", tip: "Batting Avg on Balls in Play — how often non-HR batted balls fall for hits. .300 is average." },
  woba:  { label: "wOBA",  tip: "Weighted On-Base Average — best single number for overall hitting. .320 is average." },
};

// Statcast stats (advanced, hidden in beginner mode)
const STATCAST_META: Record<string, { label: string; tip: string; unit?: string }> = {
  exit_velo:    { label: "Exit Velo",  tip: "Average speed the ball leaves the bat. 90+ mph is elite contact.", unit: " mph" },
  barrel_rate:  { label: "Barrel%",   tip: "% of batted balls hit at the ideal speed and angle for damage. 8%+ is great.", unit: "%" },
  k_rate:       { label: "K%",        tip: "Strikeout rate — lower is better. Under 20% means great bat-to-ball skills.", unit: "%" },
  bb_rate:      { label: "BB%",       tip: "Walk rate — higher means better plate discipline.", unit: "%" },
  sprint_speed: { label: "Sprint Spd", tip: "Running speed in ft/sec. 28+ ft/s is fast enough to beat out grounders.", unit: " ft/s" },
};

function computeStreakStatus(recentStats: PlayerStat[], seasonAvg: number | null): "hot" | "cold" | "neutral" {
  if (!recentStats || recentStats.length < 5 || !seasonAvg || seasonAvg === 0) return "neutral";
  const last5 = recentStats.slice(0, 5);
  const last5Avg = last5.reduce((sum, s) => sum + (s.batting_avg ?? 0), 0) / last5.length;
  if (last5Avg >= seasonAvg * 1.15) return "hot";
  if (last5Avg <= seasonAvg * 0.85) return "cold";
  return "neutral";
}

export default function PlayerDetailPage() {
  const params = useParams();
  const router = useRouter();
  const playerId = Number(params.playerId);

  const { data: detail, isLoading, error } = useQuery({
    queryKey: ["player", playerId],
    queryFn: () => getPlayer(playerId),
    enabled: !!playerId,
  });

  if (isLoading) {
    return (
      <div className="max-w-5xl mx-auto space-y-4">
        <div className="skeleton h-5 w-16 rounded" />
        <div className="rounded-xl p-5 flex items-center gap-5" style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}>
          <div className="skeleton w-16 h-16 rounded-full" />
          <div className="flex-1 space-y-2">
            <div className="skeleton h-6 w-40 rounded" />
            <div className="skeleton h-3 w-32 rounded" />
          </div>
        </div>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="max-w-5xl mx-auto py-12 text-center text-sm" style={{ color: "var(--color-muted)" }}>
        {error ? "Failed to load player data." : "Player not found."}
      </div>
    );
  }

  const { player, recent_stats, season_totals } = detail;

  // Batting trend chart data
  const chartData = (recent_stats ?? [])
    .slice(0, 20)
    .reverse()
    .map((s: PlayerStat, i: number) => ({
      game: i + 1,
      label: new Date(s.game_date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      avg: s.batting_avg != null ? parseFloat(s.batting_avg.toFixed(3)) : null,
      hr: s.home_runs ?? null,
      hits: s.hits ?? null,
    }));

  // Hot/cold streak
  const seasonAvg = season_totals?.batting_avg ?? null;
  const streakStatus = computeStreakStatus(recent_stats ?? [], seasonAvg as number | null);

  // Trend for quick stat display
  const recentLen = Math.min(5, recent_stats?.length ?? 0);
  const recentAvg5 = recentLen > 0
    ? (recent_stats ?? []).slice(0, recentLen).reduce((s, r) => s + (r.batting_avg ?? 0), 0) / recentLen
    : 0;
  const avgTrend = getTrend(recentAvg5, (seasonAvg as number) || 0);

  return (
    <div className="max-w-5xl mx-auto space-y-5">
      {/* Back */}
      <button
        onClick={() => router.back()}
        className="flex items-center gap-1.5 text-xs transition-colors"
        style={{ color: "var(--color-muted)" }}
        onMouseEnter={e => (e.currentTarget.style.color = "var(--color-secondary)")}
        onMouseLeave={e => (e.currentTarget.style.color = "var(--color-muted)")}
      >
        <ArrowLeft className="w-3.5 h-3.5" /> Back
      </button>

      {/* Player Header */}
      <div
        className="rounded-xl p-5 flex items-center gap-5"
        style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
      >
        <PlayerHeadshot url={player.headshot_url} name={player.name} size="lg" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-xl font-bold" style={{ color: "var(--color-text)" }}>{player.name}</h1>
            {/* Hot/Cold Streak Badge */}
            {streakStatus === "hot" && (
              <span
                className="text-[10px] font-bold px-2 py-0.5 rounded-full cursor-help"
                style={{ background: "rgba(249,115,22,0.15)", color: "#f97316" }}
                data-tooltip="Batting significantly above season average over last 5 games"
              >
                🔥 Hot Streak
              </span>
            )}
            {streakStatus === "cold" && (
              <span
                className="text-[10px] font-bold px-2 py-0.5 rounded-full cursor-help"
                style={{ background: "rgba(147,190,223,0.15)", color: "var(--color-accent)" }}
                data-tooltip="Batting significantly below season average over last 5 games"
              >
                ❄️ Slumping
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2 mt-1 text-xs" style={{ color: "var(--color-muted)" }}>
            {player.team && <span className="font-semibold" style={{ color: "var(--color-text)" }}>{player.team}</span>}
            {player.position && <span>· {player.position}</span>}
            {player.bats && <span>· Bats {player.bats}</span>}
            {player.throws && <span>/ Throws {player.throws}</span>}
            {player.age && <span>· Age {player.age}</span>}
          </div>
          {player.current_level && player.current_level !== "MLB" && (
            <span
              className="mt-1.5 inline-block text-[10px] font-medium px-2 py-0.5 rounded-full"
              style={{ background: "rgba(147,190,223,0.15)", color: "var(--color-accent)", border: "1px solid var(--color-border)" }}
            >
              {player.current_level}
            </span>
          )}
        </div>

        {/* Quick stats */}
        {season_totals && (
          <div className="hidden md:flex gap-6">
            {[
              { key: "batting_avg", label: "AVG", tip: "Batting Average" },
              { key: "obp",         label: "OBP", tip: "On-Base Percentage" },
              { key: "slg",         label: "SLG", tip: "Slugging Percentage" },
            ].map(({ key, label, tip }) => {
              const val = season_totals[key] != null ? Number(season_totals[key]) : null;
              return (
                <div key={key} className="text-center">
                  <div
                    className="text-[10px] uppercase font-medium mb-0.5 cursor-help"
                    data-tooltip={tip}
                    style={{ color: "var(--color-muted)", borderBottom: "1px dotted var(--color-accent)" }}
                  >
                    <span className="advanced-stat">{label}</span>
                    <span className="beginner-label">{BEGINNER_NAMES[key] || label}</span>
                  </div>
                  <div className="text-lg font-bold font-mono" style={{ color: "var(--color-primary)" }}>
                    {val != null ? val.toFixed(3) : "—"}
                  </div>
                  {val != null && (
                    <div className="mt-0.5">
                      <ContextBadge stat={key} value={val} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Recent Game Log */}
        <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
          <div
            className="flex items-center gap-2 px-4 py-3"
            style={{ background: "var(--color-panel)", borderBottom: "1px solid var(--color-border)" }}
          >
            <History className="w-4 h-4" style={{ color: "var(--color-muted)" }} />
            <h2 className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>Recent Games</h2>
          </div>
          {recent_stats && recent_stats.length > 0 ? (
            <table className="fg-table w-full">
              <thead>
                <tr>
                  <th>Date</th>
                  <th style={{ textAlign: "right" }}>H</th>
                  <th style={{ textAlign: "right" }}>HR</th>
                  <th style={{ textAlign: "right" }}>RBI</th>
                  <th style={{ textAlign: "right" }}>BB</th>
                  <th style={{ textAlign: "right" }}>
                    <abbr className="stat-abbr" data-tip="Batting average for this game">AVG</abbr>
                  </th>
                </tr>
              </thead>
              <tbody>
                {recent_stats.slice(0, 10).map((s: PlayerStat, i: number) => (
                  <tr key={i}>
                    <td style={{ color: "var(--color-muted)" }}>
                      {new Date(s.game_date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                    </td>
                    <td className="numeric">{s.hits ?? "—"}</td>
                    <td className="numeric" style={{ color: s.home_runs ? "var(--color-primary)" : undefined, fontWeight: s.home_runs ? 700 : 400 }}>
                      {s.home_runs ?? "—"}
                    </td>
                    <td className="numeric">{s.rbi ?? "—"}</td>
                    <td className="numeric">{s.walks ?? "—"}</td>
                    <td className="numeric" style={{ color: "var(--color-secondary)", fontFamily: "JetBrains Mono, monospace" }}>
                      {s.batting_avg != null ? s.batting_avg.toFixed(3) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="p-6 text-center">
              <p className="text-xs" style={{ color: "var(--color-muted)" }}>No recent game data available.</p>
            </div>
          )}
        </div>

        {/* Season Totals */}
        <div className="rounded-xl p-4" style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}>
          <div className="flex items-center gap-2 mb-3">
            <BarChart2 className="w-4 h-4" style={{ color: "var(--color-muted)" }} />
            <h2 className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>Season Stats</h2>
          </div>

          {season_totals && Object.keys(season_totals).length > 0 ? (
            <div className="space-y-2">
              {/* Core stats */}
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(CORE_STAT_META).map(([k, meta]) => {
                  const v = season_totals[k];
                  if (v == null || typeof v !== "number") return null;
                  const formattedVal = meta.format ? meta.format(v) : typeof v === "number" && v < 1 ? v.toFixed(3) : Math.round(v).toString();
                  return (
                    <div
                      key={k}
                      className="rounded-lg px-3 py-2"
                      style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)" }}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span
                          className="text-[10px] uppercase tracking-wider cursor-help"
                          data-tooltip={meta.tip}
                          style={{ color: "var(--color-muted)", borderBottom: "1px dotted var(--color-accent)" }}
                        >
                          <span className="advanced-stat">{meta.label}</span>
                          <span className="beginner-label">{BEGINNER_NAMES[k] || meta.label}</span>
                        </span>
                        <span className="text-sm font-bold font-mono" style={{ color: "var(--color-text)" }}>
                          {formattedVal}
                        </span>
                      </div>
                      <ContextBadge stat={k} value={v} />
                    </div>
                  );
                })}
              </div>

              {/* Advanced stats (hidden in beginner mode) */}
              <div className="advanced-stat">
                <div className="text-[9px] uppercase tracking-wider font-semibold mb-1.5 mt-3" style={{ color: "var(--color-subtle)" }}>
                  Advanced Stats
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(ADVANCED_STAT_META).map(([k, meta]) => {
                    const v = season_totals[k];
                    if (v == null || typeof v !== "number") return null;
                    return (
                      <div
                        key={k}
                        className="rounded-lg px-2 py-2 text-center"
                        style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)" }}
                      >
                        <div
                          className="text-[9px] uppercase tracking-wider cursor-help mb-0.5"
                          data-tooltip={meta.tip}
                          style={{ color: "var(--color-muted)", borderBottom: "1px dotted var(--color-accent)" }}
                        >
                          {meta.label}
                        </div>
                        <div className="text-sm font-bold font-mono" style={{ color: "var(--color-text)" }}>
                          {formatStatValue(k, v)}
                        </div>
                        <ContextBadge stat={k} value={v} />
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs" style={{ color: "var(--color-muted)" }}>No season totals available.</p>
          )}
        </div>
      </div>

      {/* Statcast Metrics Section (Expert / Advanced, hidden in beginner mode) */}
      {season_totals && Object.keys(STATCAST_META).some(k => season_totals[k] != null) && (
        <div
          className="rounded-xl p-4 advanced-stat"
          style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
        >
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-4 h-4" style={{ color: "var(--color-secondary)" }} />
            <h2 className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>Statcast Metrics</h2>
            <span className="text-[10px]" style={{ color: "var(--color-subtle)" }}>Season averages · percentile bars show rank among all MLB players</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Object.entries(STATCAST_META).map(([k, meta]) => {
              const v = season_totals[k];
              if (v == null || typeof v !== "number") return null;
              return (
                <div key={k} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span
                      className="text-[10px] uppercase tracking-wider cursor-help"
                      data-tooltip={meta.tip}
                      style={{ color: "var(--color-muted)", borderBottom: "1px dotted var(--color-accent)" }}
                    >
                      {meta.label}
                    </span>
                    <span className="text-sm font-bold font-mono" style={{ color: "var(--color-text)" }}>
                      {formatStatValue(k, v)}
                    </span>
                  </div>
                  <PercentileBar stat={k} value={v} />
                  <div className="flex justify-end">
                    <ContextBadge stat={k} value={v} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Batting Trend Chart */}
      {chartData.length > 1 && (
        <div
          className="rounded-xl p-4"
          style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
        >
          <div className="flex items-center gap-2 mb-3">
            <User className="w-4 h-4" style={{ color: "var(--color-muted)" }} />
            <h2 className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>
              Recent Performance Trend
            </h2>
            <span className="text-[10px]" style={{ color: "var(--color-subtle)" }}>
              Last {chartData.length} games
            </span>
            {avgTrend === "up" && (
              <span className="text-[10px] font-semibold" style={{ color: "var(--color-primary)" }}>↑ Trending up</span>
            )}
            {avgTrend === "down" && (
              <span className="text-[10px] font-semibold" style={{ color: "#f97316" }}>↓ Trending down</span>
            )}
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData} margin={{ top: 4, right: 12, bottom: 0, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(147,190,223,0.15)" />
              <XAxis dataKey="label" stroke="rgba(147,190,223,0.4)" fontSize={10} tick={{ fill: "var(--color-subtle)" }} />
              <YAxis stroke="rgba(147,190,223,0.4)" fontSize={10} tick={{ fill: "var(--color-subtle)" }} />
              <Tooltip
                contentStyle={{
                  background: "var(--color-deeper)",
                  border: "1px solid var(--color-accent)",
                  borderRadius: 8,
                  color: "var(--color-text)",
                  fontSize: 11,
                }}
              />
              <Line type="monotone" dataKey="avg"  stroke="#8ef9f3" strokeWidth={2}   dot={false} name="AVG"  connectNulls />
              <Line type="monotone" dataKey="hits" stroke="#5efc8d" strokeWidth={1.5} dot={false} name="Hits" connectNulls />
              <Line type="monotone" dataKey="hr"   stroke="#f59e0b" strokeWidth={2}   dot={{ r: 3, fill: "#f59e0b" }} name="HR" connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
