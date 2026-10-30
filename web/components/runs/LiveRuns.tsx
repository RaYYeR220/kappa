"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { API_BASE, listRuns, type LiveRunRow } from "@/lib/live/api";

import styles from "@/app/runs/runs.module.css";

/** Runs on the Kappa API, when this site is pointed at one. */
export function LiveRuns() {
  const [state, setState] = useState<{ rows: LiveRunRow[] | null; error: string | null }>({ rows: null, error: null });

  useEffect(() => {
    if (!API_BASE) return;
    const ctrl = new AbortController();
    listRuns(ctrl.signal)
      .then((rows) => setState({ rows, error: null }))
      .catch((error: unknown) => {
        if (!ctrl.signal.aborted) setState({ rows: null, error: error instanceof Error ? error.message : String(error) });
      });
    return () => ctrl.abort();
  }, []);

  if (!API_BASE) return null;
  return (
    <section className={styles.devSection} aria-labelledby="live-title">
      <h2 id="live-title">On the Kappa API</h2>
      <p>
        Runs on <span className="mono">{API_BASE}</span>. Open one to track it, settle identities and ask for a review.
      </p>
      {state.error && <p role="alert">The API did not answer: {state.error}</p>}
      {state.rows && state.rows.length === 0 && <p>No runs yet. Start one with kappa new.</p>}
      {state.rows && state.rows.length > 0 && (
        <ul className={styles.liveList}>
          {state.rows.map((r) => (
            <li key={r.id}>
              <Link href={`/live/${encodeURIComponent(r.id)}`}>
                <span className="mono">{r.id}</span>
                <span>{r.title}</span>
                <span className={styles.liveState}>{r.tracked ? "tracked" : r.confirmed ? "confirmed" : "drafted"}</span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
