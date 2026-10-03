import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RepoMind",
  description: "AI-powered codebase intelligence.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
