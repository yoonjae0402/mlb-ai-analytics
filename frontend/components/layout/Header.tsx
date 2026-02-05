"use client";
import { usePathname } from "next/navigation";
import { NAV_ITEMS } from "@/lib/constants";

export default function Header() {
  const pathname = usePathname();
  const current = NAV_ITEMS.find((item) => item.path === pathname);

  return (
    <header className="sticky top-0 z-30 bg-mlb-bg/80 backdrop-blur-md border-b border-mlb-border px-8 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-mlb-text">
            {current?.name || "MLB AI Analytics"}
          </h2>
          <p className="text-xs text-mlb-muted">
            {new Date().toLocaleDateString("en-US", {
              weekday: "long",
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </p>
        </div>
      </div>
    </header>
  );
}
