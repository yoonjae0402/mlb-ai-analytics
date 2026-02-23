"use client";
import { useQuery } from "@tanstack/react-query";
import { getDataStatus } from "@/lib/api";

export default function Footer() {
  const { data: status } = useQuery({
    queryKey: ["dataStatus"],
    queryFn: getDataStatus,
    refetchInterval: 60_000,
    retry: false,
  });

  return (
    <footer
      className="mt-auto px-6 py-4 text-[11px]"
      style={{
        background: "var(--color-darker, var(--color-dark))",
        borderTop: "1px solid var(--color-border)",
        color: "var(--color-subtle)",
      }}
    >
      <div className="max-w-screen-2xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
        {/* Data Sources */}
        <div className="flex items-center gap-4">
          <span className="font-semibold" style={{ color: "var(--color-muted)" }}>
            Data Sources:
          </span>
          <span>pybaseball (Baseball Savant / Statcast)</span>
          <span>·</span>
          <span>MLB Stats API</span>
          <span>·</span>
          <span>FanGraphs public data</span>
        </div>

        {/* Center: version */}
        <div className="flex items-center gap-3">
          <span>v2.1.0</span>
          {status?.last_updated && (
            <>
              <span>·</span>
              <span>
                Last data refresh:{" "}
                <span style={{ color: "var(--color-primary)" }}>
                  {status.last_updated}
                </span>
              </span>
            </>
          )}
        </div>

        {/* Right: disclaimer */}
        <div className="text-center sm:text-right">
          <span>For educational purposes only. Not affiliated with MLB.</span>
        </div>
      </div>
    </footer>
  );
}
