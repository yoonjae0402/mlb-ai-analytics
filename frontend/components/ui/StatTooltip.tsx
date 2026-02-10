"use client";
import { useState } from "react";
import { HelpCircle } from "lucide-react";
import { STAT_TOOLTIPS } from "@/lib/stat-helpers";

interface StatTooltipProps {
  stat: string;
  customText?: string;
}

export default function StatTooltip({ stat, customText }: StatTooltipProps) {
  const [show, setShow] = useState(false);
  const text = customText || STAT_TOOLTIPS[stat];
  if (!text) return null;

  return (
    <span
      className="relative inline-flex ml-1 cursor-help"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onClick={() => setShow(!show)}
    >
      <HelpCircle className="w-3 h-3 text-mlb-muted hover:text-mlb-blue transition-colors" />
      {show && (
        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 px-3 py-2 bg-slate-800 border border-mlb-border rounded-lg shadow-xl text-[11px] text-mlb-text leading-relaxed z-50">
          <span className="block font-semibold text-mlb-blue mb-0.5">Why it matters</span>
          {text}
          <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800" />
        </span>
      )}
    </span>
  );
}
