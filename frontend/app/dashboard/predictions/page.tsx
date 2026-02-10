"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDailyPredictions, getBestBets } from "@/lib/api";
import type { DailyPrediction } from "@/lib/api";
import PlayerHeadshot from "@/components/visuals/PlayerHeadshot";
import {
  TrendingUp, AlertCircle, ArrowUpDown, Zap,
} from "lucide-react";

const SORT_OPTIONS = [
  { value: "predicted_hr", label: "Home Runs" },
  { value: "predicted_hits", label: "Hits" },
  { value: "predicted_rbi", label: "RBI" },
  { value: "predicted_walks", label: "Walks" },
];

export default function PredictionsPage() {
  const [sortBy, setSortBy] = useState("predicted_hr");

  const { data: bestBets, isLoading: betsLoading } = useQuery({
    queryKey: ["bestBets"],
    queryFn: () => getBestBets(5),
    retry: false,
  });

  const { data, isLoading } = useQuery({
    queryKey: ["dailyPredictions", sortBy],
    queryFn: () => getDailyPredictions(sortBy, 50),
    retry: false,
  });

  const predictions = data?.predictions || [];

  return (
    <div className="max-w-7xl mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <TrendingUp className="w-5 h-5 text-mlb-red" />
        <div>
          <h1 className="text-lg font-bold text-mlb-text">Prediction Hub</h1>
          <p className="text-xs text-mlb-muted">
            AI-generated predictions for today&apos;s games
            {data?.last_updated && (
              <> &middot; Updated {new Date(data.last_updated).toLocaleString()}</>
            )}
          </p>
        </div>
      </div>

      {/* Best Bets Section */}
      <div className="bg-mlb-card border border-mlb-border rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <Zap className="w-4 h-4 text-yellow-400" />
          <h2 className="text-sm font-semibold text-mlb-text">
            Daily Best Bets
          </h2>
          <span className="text-[10px] text-mlb-muted">
            Highest-confidence predictions
          </span>
        </div>

        {betsLoading ? (
          <p className="text-xs text-mlb-muted py-4 text-center">Loading...</p>
        ) : bestBets?.predictions && bestBets.predictions.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {bestBets.predictions.map((pred: DailyPrediction, i: number) => (
              <BestBetHighlight key={pred.player_id} prediction={pred} rank={i + 1} />
            ))}
          </div>
        ) : (
          <div className="text-center py-6">
            <AlertCircle className="w-5 h-5 text-mlb-muted mx-auto mb-2" />
            <p className="text-xs text-mlb-muted">
              No predictions available. Run the scheduler or train models first.
            </p>
          </div>
        )}
      </div>

      {/* Full Predictions Table */}
      <div className="bg-mlb-card border border-mlb-border rounded-lg overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-mlb-border">
          <h2 className="text-sm font-semibold text-mlb-text">
            All Predictions ({predictions.length})
          </h2>
          <div className="flex items-center gap-2">
            <ArrowUpDown className="w-3.5 h-3.5 text-mlb-muted" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-mlb-surface border border-mlb-border rounded px-2 py-1 text-xs text-mlb-text"
            >
              {SORT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  Sort by {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-mlb-border text-[11px] font-semibold text-mlb-muted uppercase tracking-wider">
              <th className="text-left px-4 py-2">#</th>
              <th className="text-left px-3 py-2">Player</th>
              <th className="text-left px-3 py-2">Team</th>
              <th className="text-right px-3 py-2">Hits</th>
              <th className="text-right px-3 py-2">HR</th>
              <th className="text-right px-3 py-2">RBI</th>
              <th className="text-right px-4 py-2">BB</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={7} className="text-center py-8 text-xs text-mlb-muted">
                  Loading predictions...
                </td>
              </tr>
            ) : predictions.length === 0 ? (
              <tr>
                <td colSpan={7} className="text-center py-8 text-xs text-mlb-muted">
                  No predictions available
                </td>
              </tr>
            ) : (
              predictions.map((pred: DailyPrediction, i: number) => (
                <tr
                  key={pred.player_id}
                  className="border-b border-mlb-border/50 hover:bg-mlb-surface/50 transition-colors"
                >
                  <td className="px-4 py-2 text-xs text-mlb-muted">{i + 1}</td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <PlayerHeadshot
                        url={pred.headshot_url}
                        name={pred.player_name}
                        size="sm"
                      />
                      <span className="text-xs font-medium text-mlb-text">
                        {pred.player_name}
                      </span>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-xs text-mlb-muted">{pred.team || "—"}</td>
                  <td className="px-3 py-2 text-right text-xs text-mlb-text font-mono">
                    {pred.predicted_hits.toFixed(2)}
                  </td>
                  <td className="px-3 py-2 text-right text-xs font-mono font-semibold text-mlb-red">
                    {pred.predicted_hr.toFixed(3)}
                  </td>
                  <td className="px-3 py-2 text-right text-xs text-mlb-text font-mono">
                    {pred.predicted_rbi.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 text-right text-xs text-mlb-text font-mono">
                    {pred.predicted_walks.toFixed(2)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function BestBetHighlight({ prediction, rank }: { prediction: DailyPrediction; rank: number }) {
  return (
    <div className="bg-mlb-surface/50 border border-mlb-border/50 rounded-lg p-3 text-center relative">
      <span className="absolute top-1.5 left-2 text-[10px] font-bold text-mlb-red">#{rank}</span>
      <PlayerHeadshot
        url={prediction.headshot_url}
        name={prediction.player_name}
        size="md"
        className="mx-auto mb-2"
      />
      <p className="text-xs font-semibold text-mlb-text truncate">{prediction.player_name}</p>
      <p className="text-[10px] text-mlb-muted mb-2">{prediction.team}</p>
      <div className="grid grid-cols-2 gap-1 text-[10px]">
        <div className="bg-mlb-card rounded px-1 py-0.5">
          <span className="text-mlb-muted">H </span>
          <span className="text-mlb-text font-medium">{prediction.predicted_hits.toFixed(1)}</span>
        </div>
        <div className="bg-mlb-card rounded px-1 py-0.5">
          <span className="text-mlb-muted">HR </span>
          <span className="text-mlb-red font-semibold">{prediction.predicted_hr.toFixed(2)}</span>
        </div>
        <div className="bg-mlb-card rounded px-1 py-0.5">
          <span className="text-mlb-muted">RBI </span>
          <span className="text-mlb-text font-medium">{prediction.predicted_rbi.toFixed(1)}</span>
        </div>
        <div className="bg-mlb-card rounded px-1 py-0.5">
          <span className="text-mlb-muted">BB </span>
          <span className="text-mlb-text font-medium">{prediction.predicted_walks.toFixed(1)}</span>
        </div>
      </div>
    </div>
  );
}
