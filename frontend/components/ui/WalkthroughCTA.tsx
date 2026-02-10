"use client";
import { useWalkthrough } from "@/hooks/useWalkthrough";
import { Compass } from "lucide-react";

export default function WalkthroughCTA() {
  const { start, hasCompleted } = useWalkthrough();

  return (
    <button
      onClick={start}
      className="inline-flex items-center gap-2 bg-mlb-blue hover:bg-mlb-blue/80 text-white text-sm font-semibold px-5 py-2.5 rounded-lg transition-colors"
    >
      <Compass className="w-4 h-4" />
      {hasCompleted ? "Retake the Tour" : "Take a Tour"}
    </button>
  );
}
