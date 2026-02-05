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
          bg: "#0a1628",
          card: "#111d32",
          border: "#1e3050",
          red: "#e63946",
          blue: "#4895ef",
          green: "#2dc653",
          text: "#e8ecf1",
          muted: "#8899aa",
          surface: "#162240",
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
