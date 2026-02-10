"use client";
import { useQuery } from "@tanstack/react-query";
import { getDataStatus, getHealth } from "@/lib/api";
import MetricCard from "@/components/cards/MetricCard";
import {
  BarChart3, Eye, Layers, Activity, Search, FileCode, Calendar,
  TrendingUp, Users,
} from "lucide-react";
import Link from "next/link";

const features = [
  {
    title: "Scores & Schedule",
    desc: "Live games, scores, and win probability tracking",
    href: "/dashboard",
    icon: Activity,
    color: "text-mlb-green",
  },
  {
    title: "Prediction Hub",
    desc: "AI-generated daily predictions with best bets",
    href: "/dashboard/predictions",
    icon: TrendingUp,
    color: "text-mlb-red",
  },
  {
    title: "Schedule Calendar",
    desc: "Weekly and monthly game calendar with matchups",
    href: "/dashboard/schedule",
    icon: Calendar,
    color: "text-mlb-blue",
  },
  {
    title: "Player Index",
    desc: "Searchable index of all MLB and MiLB players",
    href: "/dashboard/players",
    icon: Users,
    color: "text-purple-400",
  },
  {
    title: "Player Predict",
    desc: "Search any player and predict their next game",
    href: "/predict",
    icon: Search,
    color: "text-yellow-400",
  },
  {
    title: "Model Comparison",
    desc: "Train LSTM & XGBoost side-by-side on real data",
    href: "/models",
    icon: BarChart3,
    color: "text-orange-400",
  },
  {
    title: "Attention Viz",
    desc: "Inspect what the neural network focuses on",
    href: "/attention",
    icon: Eye,
    color: "text-cyan-400",
  },
  {
    title: "Ensemble Lab",
    desc: "Experiment with model combination strategies",
    href: "/ensemble",
    icon: Layers,
    color: "text-green-400",
  },
  {
    title: "Architecture",
    desc: "System design, pipeline docs, and API reference",
    href: "/architecture",
    icon: FileCode,
    color: "text-mlb-muted",
  },
];

export default function HomePage() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: false,
  });
  const { data: status } = useQuery({
    queryKey: ["dataStatus"],
    queryFn: getDataStatus,
    retry: false,
  });

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Hero */}
      <div className="py-6">
        <h1 className="text-2xl font-bold text-mlb-text">
          <span className="text-mlb-red">MLB</span> AI Analytics
        </h1>
        <p className="text-sm text-mlb-muted mt-1 max-w-xl">
          Professional baseball analytics with AI-powered predictions.
          Real Statcast data, MiLB coverage, and automated daily projections.
        </p>
        <div className="flex items-center gap-2 mt-3">
          <span
            className={`w-2 h-2 rounded-full ${
              health?.status === "ok" ? "bg-green-400" : "bg-mlb-muted"
            }`}
          />
          <span className="text-[10px] text-mlb-muted">
            API {health?.status === "ok" ? "Connected" : "Offline"}
          </span>
        </div>
      </div>

      {/* Quick Stats */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard
            label="Players Tracked"
            value={status.players_count}
            delta={`${status.seasons.length} seasons`}
          />
          <MetricCard
            label="Game Stats"
            value={status.stats_count.toLocaleString()}
            delta="Individual game records"
          />
          <MetricCard
            label="Predictions Made"
            value={status.predictions_count}
          />
          <MetricCard
            label="Last Updated"
            value={status.last_updated || "N/A"}
          />
        </div>
      )}

      {/* Feature Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {features.map((feat) => {
          const Icon = feat.icon;
          return (
            <Link
              key={feat.href}
              href={feat.href}
              className="bg-mlb-card border border-mlb-border rounded-lg p-4 hover:border-mlb-blue/40 transition-colors group"
            >
              <Icon className={`w-5 h-5 ${feat.color} mb-2`} />
              <h3 className="text-sm font-semibold text-mlb-text group-hover:text-mlb-blue transition-colors">
                {feat.title}
              </h3>
              <p className="text-[11px] text-mlb-muted mt-1">{feat.desc}</p>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
