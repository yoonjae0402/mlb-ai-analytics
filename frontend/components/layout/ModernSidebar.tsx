"use client";
import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home, BarChart3, Eye, Layers, Activity, Search, FileCode,
  Calendar, TrendingUp, Users, ChevronLeft, ChevronRight, Scale,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { NAV_SECTIONS } from "@/lib/constants";

const ICONS: Record<string, React.ElementType> = {
  Home, BarChart3, Eye, Layers, Activity, Search, FileCode,
  Calendar, TrendingUp, Users, ChevronLeft, ChevronRight, Scale,
};

export default function ModernSidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen bg-mlb-card border-r border-mlb-border flex flex-col transition-all duration-200",
        collapsed ? "w-16" : "w-60"
      )}
    >
      {/* Header */}
      <div className="p-4 border-b border-mlb-border flex items-center justify-between">
        {!collapsed && (
          <div>
            <h1 className="text-lg font-bold text-mlb-text">
              <span className="text-mlb-red">MLB</span> AI
            </h1>
            <p className="text-[10px] text-mlb-muted mt-0.5">Pro Analytics</p>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 rounded-md hover:bg-mlb-surface text-mlb-muted hover:text-mlb-text transition-colors"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-2 overflow-y-auto">
        {NAV_SECTIONS.map((section) => (
          <div key={section.label} className="mb-1">
            {!collapsed && (
              <p className="px-4 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-mlb-muted/60">
                {section.label}
              </p>
            )}
            {section.items.map((item) => {
              const Icon = ICONS[item.icon] || Home;
              const isActive = pathname === item.path;

              return (
                <Link
                  key={item.path}
                  href={item.path}
                  title={collapsed ? item.name : undefined}
                  className={cn(
                    "flex items-center gap-2.5 mx-2 px-2.5 py-2 rounded-md text-[13px] transition-colors",
                    isActive
                      ? "bg-mlb-red/10 text-mlb-red font-medium"
                      : "text-mlb-muted hover:text-mlb-text hover:bg-mlb-surface"
                  )}
                >
                  <Icon className="w-4 h-4 flex-shrink-0" />
                  {!collapsed && <span>{item.name}</span>}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-3 border-t border-mlb-border">
        {!collapsed ? (
          <div className="text-[10px] text-mlb-muted">
            v2.1 — Pro Analytics Dashboard
          </div>
        ) : (
          <div className="text-[10px] text-mlb-muted text-center">v2.1</div>
        )}
      </div>
    </aside>
  );
}
