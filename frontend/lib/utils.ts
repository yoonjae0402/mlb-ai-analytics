import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatStat(value: number | undefined | null, decimals = 3): string {
  if (value === undefined || value === null) return "—";
  return value.toFixed(decimals);
}

export function formatPct(value: number | undefined | null): string {
  if (value === undefined || value === null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function formatMetric(value: number, precision = 4): string {
  return value.toFixed(precision);
}
