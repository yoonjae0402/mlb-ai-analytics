"use client";
import { useState } from "react";
import { User } from "lucide-react";
import { cn } from "@/lib/utils";

interface PlayerHeadshotProps {
  url?: string | null;
  name?: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const SIZES = {
  sm: "w-8 h-8",
  md: "w-12 h-12",
  lg: "w-20 h-20",
};

const ICON_SIZES = {
  sm: "w-4 h-4",
  md: "w-6 h-6",
  lg: "w-10 h-10",
};

export default function PlayerHeadshot({ url, name, size = "md", className }: PlayerHeadshotProps) {
  const [hasError, setHasError] = useState(false);

  if (!url || hasError) {
    return (
      <div
        className={cn(
          "rounded-full bg-mlb-surface flex items-center justify-center border border-mlb-border",
          SIZES[size],
          className
        )}
        title={name}
      >
        <User className={cn("text-mlb-muted", ICON_SIZES[size])} />
      </div>
    );
  }

  return (
    <img
      src={url}
      alt={name || "Player headshot"}
      onError={() => setHasError(true)}
      className={cn("rounded-full object-cover border border-mlb-border", SIZES[size], className)}
    />
  );
}
