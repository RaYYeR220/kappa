import Link from "next/link";

import { Nav } from "@/components/site/Nav";

export default function NotFound() {
  return (
    <>
      <Nav />
      <main id="main" className="wrap" style={{ padding: "96px var(--gutter) 160px", maxWidth: 760 }}>
        <h1 style={{ font: "500 44px/1.05 var(--display)", letterSpacing: "-0.035em" }}>Nothing recorded here.</h1>
        <p style={{ marginTop: 16, fontSize: 17, color: "var(--g6)" }}>
          The page or run you asked for does not exist. Recorded runs are listed on the runs page.
        </p>
        <p style={{ marginTop: 28 }}>
          <Link href="/runs" style={{ color: "var(--accent)", fontWeight: 500, borderBottom: "1px solid currentColor" }}>
            See recorded runs
          </Link>
        </p>
      </main>
    </>
  );
}
