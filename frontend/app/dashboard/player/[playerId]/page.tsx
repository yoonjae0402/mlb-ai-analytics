"use client";
import { useParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import {
  getPlayer,
  getPlayerPredictions,
  type PlayerStat,
  type PredictionRecord,
} from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import ContextBadge from "@/components/ui/ContextBadge";
import { formatStatValue } from "@/lib/stat-helpers";
import { ArrowLeft, TrendingUp, BarChart2, History } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export default function PlayerDetailPage() {
  const params = useParams();
  const router = useRouter();
  const playerId = Number(params.playerId);

  const { data: detail, isLoading: loadingPlayer } = useQuery({
    queryKey: ["player", playerId],
    queryFn: () => getPlayer(playerId),
    enabled: !!playerId,
  });

  const { data: predHistory, isLoading: loadingPreds } = useQuery({
    queryKey: ["playerPredictions", playerId],
    queryFn: () => getPlayerPredictions(playerId),
    enabled: !!playerId,
  });

  if (loadingPlayer) {
    return (
      <div className="max-w-5xl mx-auto py-12 text-center text-mlb-muted text-sm">
        Loading player...
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="max-w-5xl mx-auto py-12 text-center text-mlb-muted text-sm">
        Player not found.
      </div>
    );
  }

  const { player, recent_stats, season_totals } = detail;

  // Format prediction history for chart
  const predChartData = (predHistory ?? [])
    .slice(-20)
    .map((p: PredictionRecord, i: number) => ({
      game: i + 1,
      hits: +p.predicted_hits.toFixed(2),
      hr: +p.predicted_hr.toFixed(3),
      rbi: +p.predicted_rbi.toFixed(2),
      walks: +p.predicted_walks.toFixed(2),
      date: new Date(p.created_at).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }),
    }));

  return (
    <div className="max-w-5xl mx-auto space-y-5">
      {/* Back button */}
      <button
        onClick={() => router.back()}
        className="flex items-center gap-1.5 text-xs text-mlb-muted hover:text-mlb-text transition-colors"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        Back
      </button>

      {/* Player Header */}
      <div className="bg-mlb-card border border-mlb-border rounded-xl p-5 flex items-center gap-5">
        <PlayerHeadshot
          url={player.headshot_url}
          name={player.name}
          size="lg"
        />
        <div className="flex-1 min-w-0">
          <h1 className="text-xl font-bold text-mlb-text">{player.name}</h1>
          <div className="flex items-center gap-3 mt-1 text-xs text-mlb-muted">
            {player.team && <span className="font-semibold text-mlb-text">{player.team}</span>}
            {player.position && <span>· {player.position}</span>}
            {player.bats && <span>· Bats {player.bats}</span>}
            {player.throws && <span>Throws {player.throws}</span>}
            {player.age && <span>· Age {player.age}</span>}
          </div>
          {player.current_level && player.current_level !== "MLB" && (
            <span className="mt-1 inline-block text-[10px] bg-mlb-surface border border-mlb-border px-2 py-0.5 rounded-full text-mlb-muted">
              {player.current_level}
            </span>
          )}
        </div>

        {/* Season totals quick stats */}
        {season_totals && (
          <div className="hidden md:grid grid-cols-3 gap-4 text-center">
            {[
              { key: "avg", label: "AVG", format: (v: number) => v?.toFixed(3) },
              { key: "obp", label: "OBP", format: (v: number) => v?.toFixed(3) },
              { key: "slg", label: "SLG", format: (v: number) => v?.toFixed(3) },
            ].map(({ key, label, format }) => (
              <div key={key}>
                <p className="text-[10px] text-mlb-muted uppercase">{label}</p>
                <p className="text-lg font-bold text-mlb-text font-mono">
                  {season_totals[key] != null ? format(season_totals[key]) : "—"}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Recent game log */}
        <div className="bg-mlb-card border border-mlb-border rounded-xl overflow-hidden">
          <div className="flex items-center gap-2 px-4 py-3 border-b border-mlb-border">
            <History className="w-4 h-4 text-mlb-muted" />
            <h2 className="text-sm font-semibold text-mlb-text">Recent Games</h2>
          </div>
          {recent_stats && recent_stats.length > 0 ? (
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-mlb-border text-mlb-muted">
                  <th className="text-left px-4 py-2">Date</th>
                  <th className="text-right px-2 py-2">H</th>
                  <th className="text-right px-2 py-2">HR</th>
                  <th className="text-right px-2 py-2">RBI</th>
                  <th className="text-right px-2 py-2">BB</th>
                  <th className="text-right px-3 py-2">AVG</th>
                </tr>
              </thead>
              <tbody>
                {recent_stats.slice(0, 10).map((s: PlayerStat, i: number) => (
                  <tr
                    key={i}
                    className="border-b border-mlb-border/30 hover:bg-mlb-surface/30"
                  >
                    <td className="px-4 py-1.5 text-mlb-muted">
                      {new Date(s.game_date).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                      })}
                    </td>
                    <td className="text-right px-2 py-1.5 text-mlb-text font-mono">
                      {s.hits ?? "—"}
                    </td>
                    <td className="text-right px-2 py-1.5 text-mlb-red font-mono font-semibold">
                      {s.home_runs ?? "—"}
                    </td>
                    <td className="text-right px-2 py-1.5 text-mlb-text font-mono">
                      {s.rbi ?? "—"}
                    </td>
                    <td className="text-right px-2 py-1.5 text-mlb-text font-mono">
                      {s.walks ?? "—"}
                    </td>
                    <td className="text-right px-3 py-1.5">
                      {s.batting_avg != null ? (
                        <span className="font-mono text-mlb-text">
                          {s.batting_avg.toFixed(3)}
                          <ContextBadge
                            stat="predicted_hits"
                            value={s.batting_avg * 3}
                          />
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-xs text-mlb-muted p-4">No recent game data.</p>
          )}
        </div>

        {/* Season Totals */}
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <BarChart2 className="w-4 h-4 text-mlb-muted" />
            <h2 className="text-sm font-semibold text-mlb-text">Season Totals</h2>
          </div>
          {season_totals && Object.keys(season_totals).length > 0 ? (
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(season_totals).map(([k, v]) => {
                if (v == null || typeof v !== "number") return null;
                return (
                  <div
                    key={k}
                    className="bg-mlb-surface rounded-lg px-3 py-2 flex justify-between items-center"
                  >
                    <span className="text-[10px] text-mlb-muted uppercase tracking-wider">
                      {k.replace(/_/g, " ")}
                    </span>
                    <span className="text-sm font-bold text-mlb-text font-mono">
                      {typeof v === "number" && v < 1
                        ? v.toFixed(3)
                        : Math.round(v as number)}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-mlb-muted">No season totals available.</p>
          )}
        </div>
      </div>

      {/* Prediction History Chart */}
      {!loadingPreds && predChartData.length > 0 && (
        <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-mlb-muted" />
            <h2 className="text-sm font-semibold text-mlb-text">
              Prediction History
            </h2>
            <span className="text-[10px] text-mlb-muted ml-1">
              Last {predChartData.length} predictions
            </span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={predChartData} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3050" />
              <XAxis dataKey="date" stroke="#8899aa" fontSize={10} tick={{ fontSize: 10 }} />
              <YAxis stroke="#8899aa" fontSize={10} tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#111d32",
                  border: "1px solid #1e3050",
                  borderRadius: 8,
                  color: "#e8ecf1",
                  fontSize: 11,
                }}
              />
              <Line
                type="monotone"
                dataKey="hits"
                stroke="#4895ef"
                strokeWidth={2}
                dot={false}
                name="Pred Hits"
              />
              <Line
                type="monotone"
                dataKey="rbi"
                stroke="#2dc653"
                strokeWidth={2}
                dot={false}
                name="Pred RBI"
              />
              <Line
                type="monotone"
                dataKey="hr"
                stroke="#e63946"
                strokeWidth={1.5}
                dot={false}
                name="Pred HR"
                strokeDasharray="4 2"
              />
            </LineChart>
          </ResponsiveContainer>

          {/* Sparkline stat badges */}
          <div className="flex gap-3 mt-3">
            {[
              { label: "Avg Pred Hits", value: predChartData.reduce((s, d) => s + d.hits, 0) / predChartData.length, stat: "predicted_hits" },
              { label: "Avg Pred HR", value: predChartData.reduce((s, d) => s + d.hr, 0) / predChartData.length, stat: "predicted_hr" },
              { label: "Avg Pred RBI", value: predChartData.reduce((s, d) => s + d.rbi, 0) / predChartData.length, stat: "predicted_rbi" },
            ].map(({ label, value, stat }) => (
              <div
                key={label}
                className="flex-1 bg-mlb-surface rounded-lg p-2 text-center"
              >
                <p className="text-[10px] text-mlb-muted">{label}</p>
                <p className="text-base font-bold text-mlb-text font-mono">
                  {formatStatValue(stat, value)}
                </p>
                <ContextBadge stat={stat} value={value} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
