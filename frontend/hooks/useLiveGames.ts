"use client";
import { useQuery } from "@tanstack/react-query";
import { getLiveGames } from "@/lib/api";

export function useLiveGames() {
  return useQuery({
    queryKey: ["liveGames"],
    queryFn: getLiveGames,
    refetchInterval: 30000, // 30 seconds
  });
}
