import type { Metadata } from "next";
import "./globals.css";
import ModernSidebar from "@/components/layout/ModernSidebar";
import Header from "@/components/layout/Header";
import Providers from "./providers";
import GuidedWalkthrough from "@/components/ui/GuidedWalkthrough";

export const metadata: Metadata = {
  title: "MLB AI Analytics — Pro Dashboard",
  description: "Professional MLB analytics platform with AI-powered predictions",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <div className="flex min-h-screen">
            <ModernSidebar />
            <div className="flex-1 ml-60">
              <Header />
              <main className="p-6">{children}</main>
            </div>
          </div>
          <GuidedWalkthrough />
        </Providers>
      </body>
    </html>
  );
}
