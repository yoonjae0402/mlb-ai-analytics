"use client";
import { useQuery } from "@tanstack/react-query";
import { getDataStatus, getHealth } from "@/lib/api";
import Link from "next/link";
import {
  Activity, Calendar, Users, Trophy, BarChart3, Eye, Layers,
  Sliders, Cpu, TrendingUp, Zap,
} from "lucide-react";

const FEATURE_CARDS = [
  {
    title: "Today's Games",
    desc: "Live scores, win probability graphs, starting lineups, and projected runs for every game on the slate.",
    href: "/dashboard",
    icon: Activity,
    color: "var(--color-primary)",
    badge: "Live",
  },
  {
    title: "Schedule",
    desc: "Week and month calendar view with game times, probable pitchers, and statistical analysis.",
    href: "/dashboard/schedule",
    icon: Calendar,
    color: "var(--color-secondary)",
  },
  {
    title: "Player Index",
    desc: "Browse every MLB player. Filter by team, position, or search by name to view detailed stats.",
    href: "/dashboard/players",
    icon: Users,
    color: "var(--color-accent)",
  },
  {
    title: "Pitcher Stats",
    desc: "Search any pitcher and view ERA, WHIP, K/9, BB/9 with context ratings (Elite / Great / Average).",
    href: "/dashboard/pitchers",
    icon: TrendingUp,
    color: "var(--color-accent)",
  },
  {
    title: "Compare Players",
    desc: "Side-by-side stat comparison between two players with percentile bars and trend indicators.",
    href: "/dashboard/compare",
    icon: Zap,
    color: "var(--color-secondary)",
  },
  {
    title: "Leaderboard",
    desc: "Top performers ranked by composite score based on recent stats and projected performance.",
    href: "/dashboard/leaderboard",
    icon: Trophy,
    color: "var(--color-primary)",
  },
];

const ANALYSIS_CARDS = [
  {
    title: "Model Comparison",
    desc: "Train LSTM, XGBoost, LightGBM, and Ridge regression side-by-side. Compare metrics live.",
    href: "/models",
    icon: BarChart3,
  },
  {
    title: "Attention Visualizer",
    desc: "Inspect which game-by-game stats the LSTM neural network focuses on most for its predictions.",
    href: "/attention",
    icon: Eye,
  },
  {
    title: "Ensemble Lab",
    desc: "Experiment with weighted average vs stacking meta-learner combination strategies.",
    href: "/ensemble",
    icon: Layers,
  },
  {
    title: "Hyperparameter Tuning",
    desc: "Optuna Bayesian optimization for LSTM and XGBoost — view trial history and best parameters.",
    href: "/tuning",
    icon: Sliders,
  },
];

const STAT_GLOSSARY = [
  { abbr: "AVG", full: "Batting Average", tip: "Hits divided by at-bats. A .300 average is considered excellent in the modern era." },
  { abbr: "OBP", full: "On-Base Percentage", tip: "How often a batter reaches base (hits + walks + HBP). More valuable than AVG." },
  { abbr: "SLG", full: "Slugging Percentage", tip: "Total bases divided by at-bats. Measures power hitting." },
  { abbr: "OPS", full: "On-Base + Slugging", tip: "OBP + SLG combined. A .900+ OPS is elite." },
  { abbr: "wOBA", full: "Weighted On-Base Average", tip: "Like OBP but weighs each way of reaching base by its run value. The best single offensive stat." },
  { abbr: "ERA", full: "Earned Run Average", tip: "Average earned runs allowed per 9 innings. Lower is better. Under 3.00 is ace territory." },
  { abbr: "WHIP", full: "Walks + Hits per Inning Pitched", tip: "How many runners per inning a pitcher allows. Under 1.20 is excellent." },
  { abbr: "K/9", full: "Strikeouts per 9 Innings", tip: "Strikeout rate. Above 10 K/9 is elite for a starter." },
];

function SkeletonCard() {
  return (
    <div className="fg-card p-4 space-y-2">
      <div className="skeleton h-4 w-16 rounded" />
      <div className="skeleton h-6 w-24 rounded" />
      <div className="skeleton h-3 w-32 rounded" />
    </div>
  );
}

export default function HomePage() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: false,
  });
  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ["dataStatus"],
    queryFn: getDataStatus,
    retry: false,
  });

  const isConnected = health?.status === "ok";

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* ── Hero ──────────────────────────────────────── */}
      <div
        className="rounded-xl p-8 relative overflow-hidden"
        style={{
          background: "linear-gradient(135deg, var(--color-panel) 0%, var(--color-dark) 100%)",
          border: "1px solid var(--color-border)",
        }}
      >
        {/* Decorative background element */}
        <div
          className="absolute top-0 right-0 w-64 h-64 rounded-full opacity-10 pointer-events-none"
          style={{
            background: "var(--color-primary)",
            transform: "translate(30%, -30%)",
          }}
        />

        <div className="relative">
          <div className="flex items-center gap-2 mb-3">
            <span
              className="flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-full"
              style={{ background: "rgba(94,252,141,0.15)", color: "var(--color-primary)", border: "1px solid rgba(94,252,141,0.3)" }}
            >
              <span
                className="w-1.5 h-1.5 rounded-full live-dot"
                style={{ background: "var(--color-primary)" }}
              />
              {isConnected ? "API Connected · Real MLB Data" : "API Offline"}
            </span>
          </div>

          <h1 className="text-3xl font-bold mb-2">
            <span style={{ color: "var(--color-primary)" }}>MLB</span>{" "}
            <span style={{ color: "var(--color-text)" }}>Baseball Analytics</span>
          </h1>
          <p className="text-sm max-w-2xl" style={{ color: "var(--color-muted)" }}>
            Professional-grade baseball statistics powered by real Statcast data, machine learning win probability,
            and a beginner-friendly design that explains every stat in plain English. No prior baseball knowledge required.
          </p>

          <div className="flex flex-wrap gap-3 mt-5">
            <Link
              href="/dashboard"
              className="px-5 py-2.5 rounded-lg text-sm font-semibold transition-all"
              style={{ background: "var(--color-primary)", color: "#1a1a2e" }}
              onMouseEnter={e => (e.currentTarget.style.opacity = "0.9")}
              onMouseLeave={e => (e.currentTarget.style.opacity = "1")}
            >
              View Today&apos;s Games
            </Link>
            <Link
              href="/dashboard/players"
              className="px-5 py-2.5 rounded-lg text-sm font-medium transition-all"
              style={{
                background: "transparent",
                color: "var(--color-secondary)",
                border: "1px solid var(--color-secondary)",
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = "rgba(142,249,243,0.1)";
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = "transparent";
              }}
            >
              Browse Players
            </Link>
          </div>
        </div>
      </div>

      {/* ── Quick Stats ───────────────────────────────── */}
      <div>
        <h2 className="text-sm font-semibold uppercase tracking-wider mb-3" style={{ color: "var(--color-muted)" }}>
          Platform Stats
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {statusLoading ? (
            Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)
          ) : (
            <>
              <QuickStat label="Players Tracked" value={status?.players_count?.toLocaleString() ?? "—"} sublabel={`${status?.seasons?.length ?? 0} seasons`} />
              <QuickStat label="Game Records" value={status?.stats_count?.toLocaleString() ?? "—"} sublabel="Individual game logs" />
              <QuickStat label="Predictions Made" value={status?.predictions_count?.toLocaleString() ?? "—"} sublabel="AI-generated forecasts" />
              <QuickStat label="Last Refresh" value={status?.last_updated ?? "—"} sublabel="Auto-refreshes daily" />
            </>
          )}
        </div>
      </div>

      {/* ── Main Features ─────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider" style={{ color: "var(--color-muted)" }}>
            Main Features
          </h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {FEATURE_CARDS.map((card) => {
            const Icon = card.icon;
            return (
              <Link
                key={card.href}
                href={card.href}
                className="fg-card p-4 block group transition-all"
                style={{ textDecoration: "none" }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = card.color;
                  (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
                  (e.currentTarget as HTMLElement).style.boxShadow = `0 4px 16px rgba(0,0,0,0.2)`;
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = "var(--color-border)";
                  (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
                  (e.currentTarget as HTMLElement).style.boxShadow = "none";
                }}
              >
                <div className="flex items-start justify-between mb-2">
                  <Icon className="w-4 h-4 mt-0.5" style={{ color: card.color }} />
                  {card.badge && (
                    <span
                      className="text-[9px] font-bold px-1.5 py-0.5 rounded"
                      style={{ background: "rgba(94,252,141,0.15)", color: "var(--color-primary)" }}
                    >
                      {card.badge}
                    </span>
                  )}
                </div>
                <h3 className="text-sm font-semibold mb-1" style={{ color: "var(--color-text)" }}>
                  {card.title}
                </h3>
                <p className="text-[11px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
                  {card.desc}
                </p>
              </Link>
            );
          })}
        </div>
      </div>

      {/* ── Analysis Tools ────────────────────────────── */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4" style={{ color: "var(--color-accent)" }} />
          <h2 className="text-sm font-semibold uppercase tracking-wider" style={{ color: "var(--color-muted)" }}>
            ML Analysis Tools
          </h2>
          <span
            className="text-[9px] px-2 py-0.5 rounded-full"
            style={{ background: "rgba(131,119,209,0.2)", color: "var(--color-accent)" }}
          >
            Advanced
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {ANALYSIS_CARDS.map((card) => {
            const Icon = card.icon;
            return (
              <Link
                key={card.href}
                href={card.href}
                className="fg-card p-4 block group transition-all"
                style={{ textDecoration: "none" }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = "var(--color-accent)";
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = "var(--color-border)";
                }}
              >
                <Icon className="w-4 h-4 mb-2" style={{ color: "var(--color-accent)" }} />
                <h3 className="text-xs font-semibold mb-1" style={{ color: "var(--color-text)" }}>
                  {card.title}
                </h3>
                <p className="text-[11px] leading-relaxed" style={{ color: "var(--color-muted)" }}>
                  {card.desc}
                </p>
              </Link>
            );
          })}
        </div>
      </div>

      {/* ── Beginner Stat Glossary ────────────────────── */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider" style={{ color: "var(--color-muted)" }}>
            Stat Glossary
          </h2>
          <span
            className="text-[9px] px-2 py-0.5 rounded-full"
            style={{ background: "rgba(94,252,141,0.1)", color: "var(--color-primary)" }}
          >
            Hover for explanation
          </span>
        </div>
        <div
          className="rounded-xl p-4"
          style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
        >
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {STAT_GLOSSARY.map((stat) => (
              <div
                key={stat.abbr}
                className="p-3 rounded-lg cursor-help"
                data-tooltip={stat.tip}
                style={{
                  background: "var(--color-dark)",
                  border: "1px solid var(--color-border)",
                  transition: "border-color 0.15s",
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = "var(--color-accent)";
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = "var(--color-border)";
                }}
              >
                <div className="text-sm font-bold mb-0.5" style={{ color: "var(--color-primary)" }}>
                  {stat.abbr}
                </div>
                <div className="text-[10px]" style={{ color: "var(--color-muted)" }}>
                  {stat.full}
                </div>
              </div>
            ))}
          </div>
          <p className="text-[10px] mt-3" style={{ color: "var(--color-subtle)" }}>
            Enable <strong style={{ color: "var(--color-primary)" }}>Beginner Mode</strong> in the top navigation to see plain-English labels and hide advanced stats across the entire site.
          </p>
        </div>
      </div>
    </div>
  );
}

function QuickStat({ label, value, sublabel }: { label: string; value: string; sublabel?: string }) {
  return (
    <div
      className="rounded-lg p-4"
      style={{ background: "var(--color-card)", border: "1px solid var(--color-border)" }}
    >
      <div className="text-[10px] font-medium uppercase tracking-wider mb-1" style={{ color: "var(--color-muted)" }}>
        {label}
      </div>
      <div className="text-xl font-bold" style={{ color: "var(--color-primary)" }}>
        {value}
      </div>
      {sublabel && (
        <div className="text-[10px] mt-0.5" style={{ color: "var(--color-subtle)" }}>
          {sublabel}
        </div>
      )}
    </div>
  );
}
