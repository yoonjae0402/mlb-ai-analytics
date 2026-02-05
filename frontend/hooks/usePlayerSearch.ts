"use client";
import { useQuery } from "@tanstack/react-query";
import { searchPlayers } from "@/lib/api";
import { useState, useEffect } from "react";

export function usePlayerSearch() {
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), 300);
    return () => clearTimeout(timer);
  }, [query]);

  const results = useQuery({
    queryKey: ["playerSearch", debouncedQuery],
    queryFn: () => searchPlayers(debouncedQuery),
    enabled: debouncedQuery.length >= 2,
  });

  return { query, setQuery, ...results };
}
