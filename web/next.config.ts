import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // the app's own folder is the workspace root, whatever lockfiles sit above it
  turbopack: { root: process.cwd() },
  async headers() {
    return [
      {
        // Recorded runs never change once exported; the run id is the cache key.
        source: "/runs/:id/:file(video.mp4|sam.txt|sam-repair-.*\.txt|tracks.json)",
        headers: [{ key: "Cache-Control", value: "public, max-age=3600, stale-while-revalidate=86400" }],
      },
    ];
  },
};

export default nextConfig;
