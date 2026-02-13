"use client";
import {
  Radar, RadarChart as RechartsRadar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer, Legend,
} from "recharts";
import { FEATURE_DISPLAY_NAMES } from "@/lib/constants";
import ChartExplainer from "@/components/ui/ChartExplainer";

interface RadarChartProps {
  playerFeatures: number[];
  featureNames: string[];
  leagueAvg?: number[];
}

// Key features to display, in priority order across all categories
const KEY_FEATURES = [
  // Batter core
  "batting_avg", "on_base_pct", "slugging_pct", "woba",
  "barrel_rate", "exit_velocity", "hard_hit_rate",
  "k_rate", "bb_rate",
  // Pitcher matchup
  "opp_era", "opp_whip",
  // Derived / context
  "iso", "babip", "opp_quality",
];

export default function RadarChart({
  playerFeatures,
  featureNames,
  leagueAvg,
}: RadarChartProps) {
  // Pick features that exist in the data, up to 12 for readability
  const selectedIndices: number[] = [];
  for (const feat of KEY_FEATURES) {
    const idx = featureNames.indexOf(feat);
    if (idx !== -1) selectedIndices.push(idx);
    if (selectedIndices.length >= 12) break;
  }
  // Fallback if nothing matched
  if (selectedIndices.length === 0) {
    for (let i = 0; i < Math.min(featureNames.length, 12); i++) selectedIndices.push(i);
  }

  const data = selectedIndices.map((idx) => {
    const name = FEATURE_DISPLAY_NAMES[featureNames[idx]] || featureNames[idx];
    const playerVal = playerFeatures[idx] || 0;
    const leagueVal = leagueAvg ? leagueAvg[idx] : 0.5;
    const maxVal = Math.max(Math.abs(playerVal), Math.abs(leagueVal)) * 1.5 || 1;
    return {
      feature: name,
      player: (playerVal / maxVal) * 100,
      league: leagueVal ? (leagueVal / maxVal) * 100 : 50,
    };
  });

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-1">
        Player vs League Average
      </h3>
      <ChartExplainer>
        Each axis is a stat. The shape shows this player&apos;s recent performance vs. league average.
      </ChartExplainer>
      <ResponsiveContainer width="100%" height={300}>
        <RechartsRadar cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="#1e3050" />
          <PolarAngleAxis
            dataKey="feature"
            tick={{ fill: "#8899aa", fontSize: 10 }}
          />
          <PolarRadiusAxis tick={false} axisLine={false} />
          <Radar
            name="Player"
            dataKey="player"
            stroke="#e63946"
            fill="#e63946"
            fillOpacity={0.3}
          />
          <Radar
            name="League Avg"
            dataKey="league"
            stroke="#4895ef"
            fill="#4895ef"
            fillOpacity={0.1}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </RechartsRadar>
      </ResponsiveContainer>
    </div>
  );
}
