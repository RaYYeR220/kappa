import type { Metadata, Viewport } from "next";
import { Figtree, Geist_Mono } from "next/font/google";

import { Footer } from "@/components/site/Footer";

import "./globals.css";

const figtree = Figtree({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-figtree",
  display: "swap",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-geist-mono",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: {
    default: "Kappa: every number, with the frames that prove it",
    template: "%s · Kappa",
  },
  description:
    "Kappa turns any experiment video into behavioural measurements you can defend. SAM 3.1 tracks every animal; each number is verified with its frames, locked when a track fails a check, or refused.",
  openGraph: {
    title: "Kappa",
    description: "Every number, with the frames that prove it.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#f1f0ec",
  colorScheme: "light",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${figtree.variable} ${geistMono.variable}`}>
      <head>
        {/* General Sans is served by Fontshare under the ITF Free Font License, which does not allow redistributing the files. */}
        <link rel="preconnect" href="https://api.fontshare.com" crossOrigin="" />
        <link rel="preconnect" href="https://cdn.fontshare.com" crossOrigin="" />
        <link rel="stylesheet" href="https://api.fontshare.com/v2/css?f[]=general-sans@500,600&display=swap" />
      </head>
      <body>
        <a className="skip" href="#main">
          Skip to content
        </a>
        {children}
        <Footer />
        <div className="grain" aria-hidden="true" />
      </body>
    </html>
  );
}
