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
        // Legacy mlb colors (kept for backward compatibility)
        mlb: {
          bg: "#6d5a72",
          card: "#7a6580",
          border: "#93bedf",
          red: "#5efc8d",
          blue: "#8ef9f3",
          green: "#5efc8d",
          text: "#f0f4f8",
          muted: "#c8d8e8",
          surface: "#7a6580",
          gold: "#f59e0b",
          orange: "#f97316",
        },
        // New FanGraphs-inspired palette
        fg: {
          primary: "#5efc8d",    // Bright green – active states, highlights
          secondary: "#8ef9f3",  // Cyan – accents, links, hover
          accent: "#93bedf",     // Steel blue – borders, secondary elements
          panel: "#8377d1",      // Purple – nav, panels, section headers
          bg: "#6d5a72",         // Dark purple-gray – main backgrounds
          card: "#7a6580",       // Slightly lighter – cards
          surface: "#8b7090",    // Elevated surfaces
          text: "#f0f4f8",       // Primary text
          muted: "#c8d8e8",      // Secondary text
          subtle: "#a0b4c8",     // Tertiary text
          border: "#93bedf",     // Borders
          dark: "#5a4a5e",       // Deeper backgrounds
          deeper: "#4a3c50",     // Deepest backgrounds
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.3s ease-in-out",
        "slide-in": "slideIn 0.2s ease-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideIn: {
          "0%": { opacity: "0", transform: "translateX(-8px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
