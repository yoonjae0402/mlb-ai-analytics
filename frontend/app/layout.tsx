import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/layout/Sidebar";
import Header from "@/components/layout/Header";
import Providers from "./providers";

export const metadata: Metadata = {
  title: "MLB AI Analytics",
  description: "Real-time MLB player performance prediction using deep learning",
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
            <Sidebar />
            <div className="flex-1 ml-64">
              <Header />
              <main className="p-8">{children}</main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
