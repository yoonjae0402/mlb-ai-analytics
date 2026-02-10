"use client";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { getHealth } from "@/lib/api";
import { NAV_ITEMS } from "@/lib/constants";

export default function Header() {
  const pathname = usePathname();
  const current = NAV_ITEMS.find((item) => item.path === pathname);

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
  });

  const isConnected = health?.status === "ok";

  return (
    <header className="sticky top-0 z-30 bg-mlb-bg/80 backdrop-blur-md border-b border-mlb-border px-6 py-3">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-mlb-text">
            {current?.name || "MLB AI Analytics"}
          </h2>
          <p className="text-[10px] text-mlb-muted">
            {new Date().toLocaleDateString("en-US", {
              weekday: "long",
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isConnected ? "bg-green-400" : "bg-mlb-muted"
            }`}
          />
          <span className="text-[10px] text-mlb-muted">
            {isConnected ? "API Connected" : "API Offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
