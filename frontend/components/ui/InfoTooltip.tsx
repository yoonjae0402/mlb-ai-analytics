"use client";
import { useState, useRef, useEffect } from "react";
import { HelpCircle } from "lucide-react";
import glossary from "@/lib/glossary";

interface InfoTooltipProps {
  term: string;
}

export default function InfoTooltip({ term }: InfoTooltipProps) {
  const [show, setShow] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const entry = glossary[term];

  // Close on click outside
  useEffect(() => {
    if (!show) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setShow(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [show]);

  if (!entry) return null;

  return (
    <span
      ref={ref}
      className="relative inline-flex items-center ml-1"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onClick={() => setShow((s) => !s)}
    >
      <HelpCircle className="w-3.5 h-3.5 text-mlb-muted/60 hover:text-mlb-blue cursor-help transition-colors" />
      {show && (
        <span className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 bg-mlb-surface border border-mlb-border rounded-lg p-3 shadow-xl pointer-events-none">
          <span className="block text-xs font-semibold text-mlb-text mb-1">
            {entry.term}
          </span>
          <span className="block text-[11px] text-mlb-muted leading-relaxed">
            {entry.definition}
          </span>
          {/* Arrow */}
          <span className="absolute top-full left-1/2 -translate-x-1/2 w-2 h-2 bg-mlb-surface border-r border-b border-mlb-border rotate-45 -mt-1" />
        </span>
      )}
    </span>
  );
}
