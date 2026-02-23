"use client";
import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { getHealth, searchPlayers } from "@/lib/api";
import type { Player } from "@/lib/api";
import {
  Search, X, Activity, ChevronDown, BookOpen, Cpu,
} from "lucide-react";

const NAV_TABS = [
  { label: "Games",      href: "/dashboard",           exact: true  },
  { label: "Schedule",   href: "/dashboard/schedule",  exact: true  },
  { label: "Players",    href: "/dashboard/players",   exact: false },
  { label: "Pitchers",   href: "/dashboard/pitchers",  exact: true  },
  { label: "Compare",    href: "/dashboard/compare",   exact: true  },
  { label: "Leaderboard",href: "/dashboard/leaderboard", exact: true },
];

const ANALYSIS_ITEMS = [
  { label: "Model Comparison",      href: "/models"      },
  { label: "Attention Visualizer",  href: "/attention"   },
  { label: "Ensemble Lab",          href: "/ensemble"    },
  { label: "Hyperparameter Tuning", href: "/tuning"      },
  { label: "System Health",         href: "/dashboard/system" },
  { label: "Architecture",          href: "/architecture" },
];

function isTabActive(pathname: string, href: string, exact: boolean): boolean {
  if (exact) return pathname === href;
  return pathname.startsWith(href);
}

export default function TopNav() {
  const pathname = usePathname();
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Player[]>([]);
  const [searchOpen, setSearchOpen] = useState(false);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [beginnerMode, setBeginnerMode] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const analysisRef = useRef<HTMLDivElement>(null);

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
  });
  const isConnected = health?.status === "ok";

  // Beginner mode: toggle body class
  useEffect(() => {
    if (beginnerMode) {
      document.body.classList.add("beginner-mode");
    } else {
      document.body.classList.remove("beginner-mode");
    }
  }, [beginnerMode]);

  // Search debounce
  useEffect(() => {
    if (!searchQuery.trim() || searchQuery.length < 2) {
      setSearchResults([]);
      return;
    }
    const timer = setTimeout(async () => {
      try {
        const players = await searchPlayers(searchQuery);
        setSearchResults(players.slice(0, 8));
      } catch {
        setSearchResults([]);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Close dropdowns on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setSearchOpen(false);
      }
      if (analysisRef.current && !analysisRef.current.contains(e.target as Node)) {
        setAnalysisOpen(false);
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const isAnalysisActive = ANALYSIS_ITEMS.some(item => pathname.startsWith(item.href));

  return (
    <header
      className="sticky top-0 z-50 w-full"
      style={{ background: "var(--color-panel)", borderBottom: "2px solid var(--color-primary)" }}
    >
      {/* ── Top Bar ─── */}
      <div className="flex items-center justify-between px-4 py-2.5 max-w-screen-2xl mx-auto">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 flex-shrink-0">
          <div
            className="w-8 h-8 rounded flex items-center justify-center font-bold text-sm"
            style={{ background: "var(--color-primary)", color: "#1a1a2e" }}
          >
            MLB
          </div>
          <div>
            <span className="font-bold text-base" style={{ color: "var(--color-primary)" }}>
              Baseball
            </span>
            <span className="font-bold text-base" style={{ color: "var(--color-text)" }}>
              {" "}Analytics
            </span>
            <div className="text-[9px]" style={{ color: "var(--color-muted)" }}>
              Powered by AI · Real Statcast Data
            </div>
          </div>
        </Link>

        {/* Right: Search + Beginner Toggle + Status */}
        <div className="flex items-center gap-3">
          {/* Beginner Mode Toggle */}
          <button
            onClick={() => setBeginnerMode(!beginnerMode)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-all"
            style={{
              background: beginnerMode ? "var(--color-primary)" : "rgba(94,252,141,0.1)",
              color: beginnerMode ? "#1a1a2e" : "var(--color-primary)",
              border: "1px solid var(--color-primary)",
            }}
            title="Toggle Beginner Mode: shows simplified stats and plain-English explanations"
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Beginner Mode</span>
          </button>

          {/* Search */}
          <div ref={searchRef} className="relative">
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded"
              style={{ background: "var(--color-dark)", border: "1px solid var(--color-border)" }}
            >
              <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--color-muted)" }} />
              <input
                type="text"
                placeholder="Search players..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setSearchOpen(true);
                }}
                onFocus={() => setSearchOpen(true)}
                className="bg-transparent border-none outline-none text-xs w-36 sm:w-48"
                style={{ color: "var(--color-text)" }}
              />
              {searchQuery && (
                <button
                  onClick={() => { setSearchQuery(""); setSearchResults([]); }}
                  style={{ color: "var(--color-muted)" }}
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>

            {/* Search dropdown */}
            {searchOpen && searchResults.length > 0 && (
              <div
                className="absolute right-0 top-full mt-1 rounded-lg shadow-xl overflow-hidden z-50"
                style={{
                  width: "280px",
                  background: "var(--color-deeper)",
                  border: "1px solid var(--color-accent)",
                }}
              >
                {searchResults.map((player) => (
                  <button
                    key={player.id}
                    className="w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors"
                    style={{ borderBottom: "1px solid var(--color-border)" }}
                    onMouseEnter={e => (e.currentTarget.style.background = "rgba(94,252,141,0.08)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                    onClick={() => {
                      router.push(`/dashboard/player/${player.id}`);
                      setSearchOpen(false);
                      setSearchQuery("");
                    }}
                  >
                    <div
                      className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                      style={{ background: "var(--color-panel)", color: "var(--color-primary)" }}
                    >
                      {player.name.charAt(0)}
                    </div>
                    <div>
                      <div className="text-xs font-medium" style={{ color: "var(--color-text)" }}>
                        {player.name}
                      </div>
                      <div className="text-[10px]" style={{ color: "var(--color-muted)" }}>
                        {player.team} · {player.position}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* API Status */}
          <div className="flex items-center gap-1.5 hide-mobile">
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: isConnected ? "var(--color-primary)" : "var(--color-muted)" }}
            />
            <span className="text-[10px]" style={{ color: "var(--color-muted)" }}>
              {isConnected ? "Live" : "Offline"}
            </span>
          </div>
        </div>
      </div>

      {/* ── Navigation Tabs ─── */}
      <nav
        className="flex items-center overflow-x-auto px-4 max-w-screen-2xl mx-auto"
        style={{ borderTop: "1px solid rgba(147,190,223,0.2)" }}
      >
        {NAV_TABS.map((tab) => {
          const active = isTabActive(pathname, tab.href, tab.exact);
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className="nav-tab"
              style={{
                color: active ? "var(--color-primary)" : "var(--color-muted)",
                borderBottomColor: active ? "var(--color-primary)" : "transparent",
              }}
              onMouseEnter={e => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.color = "var(--color-secondary)";
                  (e.currentTarget as HTMLElement).style.borderBottomColor = "var(--color-secondary)";
                }
              }}
              onMouseLeave={e => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.color = "var(--color-muted)";
                  (e.currentTarget as HTMLElement).style.borderBottomColor = "transparent";
                }
              }}
            >
              {tab.label}
            </Link>
          );
        })}

        {/* Analysis Dropdown */}
        <div ref={analysisRef} className="relative">
          <button
            onClick={() => setAnalysisOpen(!analysisOpen)}
            className="nav-tab flex items-center gap-1"
            style={{
              color: isAnalysisActive ? "var(--color-primary)" : "var(--color-muted)",
              borderBottomColor: isAnalysisActive ? "var(--color-primary)" : "transparent",
            }}
          >
            <Cpu className="w-3.5 h-3.5" />
            Analysis
            <ChevronDown className={`w-3 h-3 transition-transform ${analysisOpen ? "rotate-180" : ""}`} />
          </button>

          {analysisOpen && (
            <div
              className="absolute left-0 top-full mt-0.5 rounded-lg shadow-xl overflow-hidden z-50"
              style={{
                minWidth: "200px",
                background: "var(--color-deeper)",
                border: "1px solid var(--color-accent)",
              }}
            >
              {ANALYSIS_ITEMS.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setAnalysisOpen(false)}
                  className="flex items-center gap-2 px-4 py-2.5 text-xs transition-colors"
                  style={{
                    color: pathname === item.href ? "var(--color-primary)" : "var(--color-muted)",
                    background: pathname === item.href ? "rgba(94,252,141,0.08)" : "transparent",
                    borderBottom: "1px solid var(--color-border)",
                  }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLElement).style.background = "rgba(94,252,141,0.08)";
                    (e.currentTarget as HTMLElement).style.color = "var(--color-secondary)";
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLElement).style.background =
                      pathname === item.href ? "rgba(94,252,141,0.08)" : "transparent";
                    (e.currentTarget as HTMLElement).style.color =
                      pathname === item.href ? "var(--color-primary)" : "var(--color-muted)";
                  }}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          )}
        </div>

        {/* Spacer + Last Updated */}
        <div className="ml-auto flex items-center gap-1.5 py-2 text-[10px] hide-mobile"
          style={{ color: "var(--color-subtle)" }}
        >
          <Activity className="w-3 h-3" />
          <span>Auto-refreshes every 30s</span>
        </div>
      </nav>
    </header>
  );
}
