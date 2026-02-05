"use client";
import { useQuery } from "@tanstack/react-query";
import { getTrainStatus } from "@/lib/api";

export function useTrainingStatus(enabled = true) {
  return useQuery({
    queryKey: ["trainStatus"],
    queryFn: getTrainStatus,
    refetchInterval: (query) =>
      query.state.data?.is_training ? 1000 : false,
    enabled,
  });
}
