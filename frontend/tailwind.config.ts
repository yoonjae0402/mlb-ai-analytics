import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        mlb: {
          bg: "#0f172a",
          card: "#1e293b",
          border: "#334155",
          red: "#e63946",
          blue: "#4895ef",
          green: "#22c55e",
          text: "#f1f5f9",
          muted: "#94a3b8",
          surface: "#1e293b",
          gold: "#f59e0b",
          orange: "#f97316",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
