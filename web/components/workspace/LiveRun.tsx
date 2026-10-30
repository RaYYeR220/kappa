"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { loadRun } from "@/lib/live/api";
import type { RunData } from "@/lib/types";

import { Workspace } from "./Workspace";
import styles from "./Workspace.module.css";

/** A run read from the live API instead of a recorded bundle. */
export function LiveRun({ id }: { id: string }) {
  const [state, setState] = useState<{ run: RunData | null; error: string | null }>({ run: null, error: null });

  useEffect(() => {
    const ctrl = new AbortController();
    loadRun(id, ctrl.signal)
      .then((run) => setState({ run, error: null }))
      .catch((error: unknown) => {
        if (!ctrl.signal.aborted) setState({ run: null, error: error instanceof Error ? error.message : String(error) });
      });
    return () => ctrl.abort();
  }, [id]);

  if (state.run) return <Workspace initial={state.run} />;
  return (
    <div className={`wrap ${styles.head}`}>
      <nav aria-label="Breadcrumb" className={styles.crumbs}>
        <Link href="/runs">Recorded runs</Link>
        <span aria-hidden="true">/</span>
        <span className="mono" aria-current="page">
          {id}
        </span>
      </nav>
      <p className={styles.empty} role={state.error ? "alert" : "status"}>
        {state.error ? `This run could not be loaded from the Kappa API: ${state.error}` : "Loading the run from the Kappa API…"}
      </p>
    </div>
  );
}
