import type { Metadata } from "next";
import "./globals.css";
import TopNav from "@/components/layout/TopNav";
import Footer from "@/components/layout/Footer";
import Providers from "./providers";

export const metadata: Metadata = {
  title: "MLB Baseball Analytics",
  description: "FanGraphs-style MLB analytics with real Statcast data, win probability, and beginner-friendly stat explanations.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ background: "var(--color-background)" }}>
        <Providers>
          <div className="flex flex-col min-h-screen">
            <TopNav />
            <main className="flex-1 px-4 py-5 max-w-screen-2xl mx-auto w-full page-enter">
              {children}
            </main>
            <Footer />
          </div>
        </Providers>
      </body>
    </html>
  );
}
