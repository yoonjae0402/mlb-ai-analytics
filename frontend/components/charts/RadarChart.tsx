"use client";
import {
  Radar, RadarChart as RechartsRadar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ResponsiveContainer, Legend,
} from "recharts";
import { FEATURE_DISPLAY_NAMES } from "@/lib/constants";

interface RadarChartProps {
  playerFeatures: number[];
  featureNames: string[];
  leagueAvg?: number[];
}

export default function RadarChart({
  playerFeatures,
  featureNames,
  leagueAvg,
}: RadarChartProps) {
  // Normalize all features to 0-100 scale
  const selectedFeatures = [0, 1, 2, 3, 4, 8, 9, 10]; // key features
  const data = selectedFeatures.map((idx) => {
    const name = FEATURE_DISPLAY_NAMES[featureNames[idx]] || featureNames[idx];
    const playerVal = playerFeatures[idx] || 0;
    const leagueVal = leagueAvg ? leagueAvg[idx] : 0.5;
    // Simple normalization
    const maxVal = Math.max(playerVal, leagueVal) * 1.5 || 1;
    return {
      feature: name,
      player: (playerVal / maxVal) * 100,
      league: leagueVal ? (leagueVal / maxVal) * 100 : 50,
    };
  });

  return (
    <div className="bg-mlb-card border border-mlb-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-mlb-text mb-4">
        Player vs League Average
      </h3>
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
