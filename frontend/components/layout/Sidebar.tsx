"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home, BarChart3, Eye, Layers, Activity, Search, FileCode,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { NAV_ITEMS } from "@/lib/constants";

const ICONS: Record<string, React.ElementType> = {
  Home, BarChart3, Eye, Layers, Activity, Search, FileCode,
};

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 bg-mlb-card border-r border-mlb-border flex flex-col">
      <div className="p-6 border-b border-mlb-border">
        <h1 className="text-xl font-bold text-mlb-text">
          <span className="text-mlb-red">MLB</span> AI Analytics
        </h1>
        <p className="text-xs text-mlb-muted mt-1">Deep Learning Platform</p>
      </div>

      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const Icon = ICONS[item.icon] || Home;
          const isActive = pathname === item.path;

          return (
            <Link
              key={item.path}
              href={item.path}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors",
                isActive
                  ? "bg-mlb-red/10 text-mlb-red border border-mlb-red/20"
                  : "text-mlb-muted hover:text-mlb-text hover:bg-mlb-surface"
              )}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {item.name}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-mlb-border">
        <div className="text-xs text-mlb-muted">
          v2.0 — Next.js + FastAPI + PostgreSQL
        </div>
      </div>
    </aside>
  );
}
