"use client";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface TeamLogoProps {
  url?: string | null;
  abbreviation?: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const SIZES = {
  sm: "w-6 h-6",
  md: "w-8 h-8",
  lg: "w-12 h-12",
};

const TEXT_SIZES = {
  sm: "text-[8px]",
  md: "text-[10px]",
  lg: "text-sm",
};

export default function TeamLogo({ url, abbreviation, size = "md", className }: TeamLogoProps) {
  const [hasError, setHasError] = useState(false);

  if (!url || hasError) {
    return (
      <div
        className={cn(
          "rounded bg-mlb-surface flex items-center justify-center border border-mlb-border font-bold text-mlb-muted",
          SIZES[size],
          TEXT_SIZES[size],
          className
        )}
      >
        {abbreviation || "?"}
      </div>
    );
  }

  return (
    <img
      src={url}
      alt={abbreviation || "Team logo"}
      onError={() => setHasError(true)}
      className={cn("object-contain", SIZES[size], className)}
    />
  );
}
