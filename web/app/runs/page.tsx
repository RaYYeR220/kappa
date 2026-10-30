import type { Metadata } from "next";
import Link from "next/link";

import { Nav } from "@/components/site/Nav";
import { LiveRuns } from "@/components/runs/LiveRuns";
import { devRunsEnabled, listRunCards } from "@/lib/bundle";
import { clockShort, frames, plural } from "@/lib/format";
import { subjectColor } from "@/lib/palette";
import type { RunCard } from "@/lib/types";

import styles from "./runs.module.css";

export const metadata: Metadata = {
  title: "Recorded runs",
  description: "Replay a Kappa run frame by frame: masks, integrity checks, and every number with its status.",
};

function Thumb({ card }: { card: RunCard }) {
  return (
    <svg viewBox={`0 0 ${card.width} ${card.height}`} className={styles.thumb} aria-hidden="true" preserveAspectRatio="xMidYMid slice">
      <rect width={card.width} height={card.height} fill="#e4e1d9" />
      {card.paths.map((p) => (
        <polyline
          key={p.subject}
          points={p.points.map(([x, y]) => `${x},${y}`).join(" ")}
          fill="none"
          stroke={subjectColor(p.subject, "light")}
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
          opacity={0.9}
        />
      ))}
      {card.paths.map((p) => {
        const last = p.points.at(-1);
        return last ? <circle key={`h${p.subject}`} cx={last[0]} cy={last[1]} r={9} fill={subjectColor(p.subject, "video")} stroke="#0e0e0c" strokeOpacity=".6" strokeWidth={2} /> : null;
      })}
    </svg>
  );
}

export default async function RunsPage() {
  const cards = await listRunCards();
  const published = cards.filter((c) => c.origin === "published");
  const dev = cards.filter((c) => c.origin === "dev");

  const list = (items: RunCard[]) => (
    <ul className={styles.list}>
      {items.map((c) => (
        <li key={c.id}>
          <Link href={`/runs/${c.id}`} className={styles.card}>
            <div className={styles.well}>
              <Thumb card={c} />
              {c.origin === "dev" && <span className={styles.badge}>Development fixture</span>}
            </div>
            <div className={styles.body}>
              <h3>{c.title}</h3>
              {c.question && <p className={styles.question}>{c.question}</p>}
              <dl className={styles.facts}>
                <div>
                  <dt>Subjects</dt>
                  <dd>
                    {c.subjectCount} {c.subjectCount === 1 ? c.subjectLabel : plural(c.subjectLabel)}
                  </dd>
                </div>
                <div>
                  <dt>Length</dt>
                  <dd>
                    {clockShort(c.durationS)}, {frames(c.frames)} frames
                  </dd>
                </div>
                <div>
                  <dt>Confirmed by</dt>
                  <dd>{c.confirmedBy ?? "not yet"}</dd>
                </div>
              </dl>
              <ul className={styles.status} aria-label="Results">
                <li data-status="verified">{c.summary.verified} verified</li>
                <li data-status="locked">
                  <span>{c.summary.locked} locked</span>
                </li>
                <li data-status="refused">{c.summary.refused} refused</li>
              </ul>
            </div>
          </Link>
        </li>
      ))}
    </ul>
  );

  return (
    <>
      <Nav current="/runs" />
      <main id="main" className={`wrap ${styles.page}`}>
        <header className={styles.head}>
          <h1>Recorded runs</h1>
          <p>
            Each run is the complete record: the video, SAM 3.1&rsquo;s raw output, the confirmed protocol, every number with
            its status, and the manifest. Open one to replay it frame by frame.
          </p>
        </header>

        {published.length > 0 ? (
          list(published)
        ) : (
          <div className={styles.empty}>
            <h2>No runs are published yet.</h2>
            <p>
              A run appears here once it is exported as a bundle into <code>public/runs/</code> and listed in{" "}
              <code>public/runs/index.json</code>. The <Link href="/docs#bundle">bundle format</Link> describes the files.
            </p>
          </div>
        )}

        {devRunsEnabled && dev.length > 0 && (
          <section className={styles.devSection} aria-labelledby="dev-title">
            <h2 id="dev-title">Development fixtures</h2>
            <p>Synthetic runs for building the workspace. Their masks are not SAM 3.1 output and their numbers are not measurements.</p>
            {list(dev)}
          </section>
        )}

        <LiveRuns />
      </main>
    </>
  );
}
