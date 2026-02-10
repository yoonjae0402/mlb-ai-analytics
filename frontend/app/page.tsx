"use client";
import { useQuery } from "@tanstack/react-query";
import { getDataStatus, getHealth } from "@/lib/api";
import MetricCard from "@/components/cards/MetricCard";
import {
  BarChart3, Eye, Layers, Activity, Search, FileCode, Database, Sparkles,
} from "lucide-react";
import Link from "next/link";
import PageIntro from "@/components/ui/PageIntro";
import WalkthroughCTA from "@/components/ui/WalkthroughCTA";

const features = [
  {
    title: "Model Comparison",
    desc: "Train LSTM & XGBoost side-by-side on real MLB data",
    href: "/models",
    icon: BarChart3,
    color: "text-mlb-red",
  },
  {
    title: "Attention Visualizer",
    desc: "Inspect what the LSTM model focuses on",
    href: "/attention",
    icon: Eye,
    color: "text-purple-400",
  },
  {
    title: "Ensemble Lab",
    desc: "Experiment with model combination strategies",
    href: "/ensemble",
    icon: Layers,
    color: "text-mlb-blue",
  },
  {
    title: "Real-Time Dashboard",
    desc: "Live MLB games with win probability tracking",
    href: "/dashboard",
    icon: Activity,
    color: "text-mlb-green",
  },
  {
    title: "Prediction Explorer",
    desc: "Search real players and predict their next game",
    href: "/predict",
    icon: Search,
    color: "text-yellow-400",
  },
  {
    title: "Architecture & Docs",
    desc: "System design, data pipeline, and API reference",
    href: "/architecture",
    icon: FileCode,
    color: "text-cyan-400",
  },
];

export default function HomePage() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
  });
  const { data: status } = useQuery({
    queryKey: ["dataStatus"],
    queryFn: getDataStatus,
  });

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <PageIntro title="What is this platform?" icon={<Sparkles className="w-5 h-5" />} pageKey="home">
        <p>
          This platform uses AI to predict how MLB players will perform in their next game,
          trained on real Statcast data (exit velocity, barrel rate, sprint speed, and more).
          No baseball or machine learning expertise needed — just explore!
        </p>
      </PageIntro>

      {/* Hero */}
      <div className="text-center py-8" data-tour="hero">
        <h1 className="text-4xl font-bold text-mlb-text">
          <span className="text-mlb-red">MLB</span> AI Analytics Platform
        </h1>
        <p className="text-mlb-muted mt-3 max-w-2xl mx-auto">
          Predicting MLB player performance using deep learning on real Statcast
          data. Powered by PyTorch, FastAPI, and Next.js.
        </p>
        <div className="flex items-center justify-center gap-2 mt-4">
          <span
            className={`w-2 h-2 rounded-full ${
              health?.status === "ok" ? "bg-mlb-green" : "bg-mlb-red"
            }`}
          />
          <span className="text-xs text-mlb-muted">
            API {health?.status === "ok" ? "Connected" : "Offline"}
          </span>
        </div>
        <div className="mt-5">
          <WalkthroughCTA />
        </div>
      </div>

      {/* Quick Stats */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" data-tour="features">
        {features.map((feat) => {
          const Icon = feat.icon;
          return (
            <Link
              key={feat.href}
              href={feat.href}
              className="bg-mlb-card border border-mlb-border rounded-xl p-5 hover:border-mlb-blue/40 transition-colors group"
            >
              <Icon className={`w-6 h-6 ${feat.color} mb-3`} />
              <h3 className="text-sm font-semibold text-mlb-text group-hover:text-mlb-blue transition-colors">
                {feat.title}
              </h3>
              <p className="text-xs text-mlb-muted mt-1">{feat.desc}</p>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
