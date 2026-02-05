"use client";
import { useMutation } from "@tanstack/react-query";
import { predictPlayer } from "@/lib/api";

export function usePrediction() {
  return useMutation({
    mutationFn: ({
      playerId,
      modelType,
    }: {
      playerId: number;
      modelType?: string;
    }) => predictPlayer(playerId, modelType),
  });
}
