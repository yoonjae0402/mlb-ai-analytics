import type { GameData } from "@/lib/api";

interface GameCardProps {
  game: GameData;
  onClick?: () => void;
}

export default function GameCard({ game, onClick }: GameCardProps) {
  const isLive = game.status === "In Progress" || game.status === "Live";
  const isFinal = game.status?.toLowerCase().includes("final");
  const isScheduled = !isLive && !isFinal;

  const homeWinPct = game.home_win_prob ?? 0.5;
  const awayWinPct = 1 - homeWinPct;

  return (
    <div
      onClick={onClick}
      className="fg-card cursor-pointer transition-all"
      style={{ textDecoration: "none" }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.borderColor = "var(--color-secondary)";
        (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
        (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 16px rgba(0,0,0,0.25)";
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.borderColor = "var(--color-border)";
        (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
        (e.currentTarget as HTMLElement).style.boxShadow = "none";
      }}
    >
      {/* Status bar */}
      <div
        className="px-3 py-1.5 flex items-center justify-between text-[10px] font-semibold"
        style={{
          background: isLive
            ? "rgba(94,252,141,0.12)"
            : isFinal
            ? "rgba(147,190,223,0.08)"
            : "rgba(131,119,209,0.15)",
          borderBottom: "1px solid var(--color-border)",
        }}
      >
        <span
          style={{
            color: isLive
              ? "var(--color-primary)"
              : isFinal
              ? "var(--color-muted)"
              : "var(--color-accent)",
          }}
        >
          {isLive ? (
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full live-dot" style={{ background: "var(--color-primary)" }} />
              LIVE · {game.half} {game.inning}
            </span>
          ) : isFinal ? (
            "FINAL"
          ) : (
            game.status || "SCHEDULED"
          )}
        </span>
        {game.venue && (
          <span
            className="truncate max-w-[110px]"
            style={{ color: "var(--color-subtle)" }}
          >
            {game.venue}
          </span>
        )}
      </div>

      <div className="p-3 space-y-2.5">
        {/* Teams + Scores */}
        <div className="space-y-1.5">
          {/* Away */}
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-bold" style={{ color: "var(--color-text)" }}>
                {game.away_abbrev}
              </span>
              {game.away_probable_pitcher && (
                <span className="text-[10px] ml-1.5" style={{ color: "var(--color-subtle)" }}>
                  {game.away_probable_pitcher}
                </span>
              )}
            </div>
            <span
              className="text-lg font-bold tabular-nums"
              style={{ color: isLive || isFinal ? "var(--color-text)" : "var(--color-subtle)" }}
            >
              {isLive || isFinal ? game.away_score : "—"}
            </span>
          </div>

          {/* Home */}
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-bold" style={{ color: "var(--color-text)" }}>
                {game.home_abbrev}
              </span>
              {game.home_probable_pitcher && (
                <span className="text-[10px] ml-1.5" style={{ color: "var(--color-subtle)" }}>
                  {game.home_probable_pitcher}
                </span>
              )}
            </div>
            <span
              className="text-lg font-bold tabular-nums"
              style={{ color: isLive || isFinal ? "var(--color-text)" : "var(--color-subtle)" }}
            >
              {isLive || isFinal ? game.home_score : "—"}
            </span>
          </div>
        </div>

        {/* Game time (scheduled) */}
        {isScheduled && game.game_datetime && (
          <div className="text-[10px]" style={{ color: "var(--color-muted)" }}>
            {new Date(game.game_datetime).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
          </div>
        )}

        {/* Win Probability Bar */}
        <div>
          <div className="flex justify-between text-[10px] mb-1" style={{ color: "var(--color-subtle)" }}>
            <span>{game.away_abbrev} {Math.round(awayWinPct * 100)}%</span>
            <span
              className="text-[9px] font-medium"
              data-tooltip="Statistical win probability based on lineup strength, starting pitcher matchup, and home field advantage"
              style={{ color: "var(--color-subtle)", cursor: "help", borderBottom: "1px dotted var(--color-accent)" }}
            >
              Win Prob
            </span>
            <span>{game.home_abbrev} {Math.round(homeWinPct * 100)}%</span>
          </div>
          <div className="win-prob-bar">
            <div className="away-side" style={{ width: `${awayWinPct * 100}%`, transition: "width 0.5s" }} />
            <div className="home-side" style={{ width: `${homeWinPct * 100}%`, transition: "width 0.5s" }} />
          </div>
          {/* Beginner verdict — only for upcoming games */}
          {isScheduled && (
            <div
              className="beginner-label text-center text-[10px] mt-1.5 font-medium"
              style={{
                color:
                  homeWinPct > 0.55
                    ? "var(--color-primary)"
                    : awayWinPct > 0.55
                    ? "var(--color-secondary)"
                    : "var(--color-muted)",
              }}
            >
              {homeWinPct > 0.65
                ? `${game.home_abbrev} favored 🏠`
                : homeWinPct > 0.55
                ? `${game.home_abbrev} slight edge`
                : awayWinPct > 0.65
                ? `${game.away_abbrev} favored`
                : awayWinPct > 0.55
                ? `${game.away_abbrev} slight edge`
                : "Toss-up 🤝"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
