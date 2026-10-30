"use client";

import { CopyButton } from "@/components/ui/CopyButton";
import { clock, eventSentence } from "@/lib/format";
import { subjectColor, subjectTag } from "@/lib/palette";
import type { EventView, RepairView } from "@/lib/types";

import type { Mark } from "./VideoStage";
import styles from "./Workspace.module.css";

export interface MarkingState {
  active: number | null;
  marks: Mark[];
  by: string;
  busy: boolean;
  message: string | null;
}

interface Props {
  runId: string;
  frame: number;
  fps: number;
  label: string;
  subjects: number[];
  blockers: EventView[];
  repair: RepairView | null;
  /** Current positions of every subject on this frame, from the tracks. */
  positions: Mark[];
  live: { canWrite: boolean; marking: MarkingState | null; onStart: () => void; onActive: (s: number) => void; onBy: (v: string) => void; onSubmit: () => void; onCancel: () => void } | null;
  onClose: () => void;
}

export function FixPanel({ runId, frame, fps, label, subjects, blockers, repair, positions, live, onClose }: Props) {
  const command = [
    `kappa repair ${runId} --frame ${frame} --by "Your Name"`,
    ...positions.map((p) => `  --subject ${p.subject}:${p.x},${p.y}`),
  ].join(" \\\n");
  const marking = live?.marking ?? null;
  const allMarked = marking ? subjects.every((s) => marking.marks.some((m) => m.subject === s)) : false;

  return (
    <section className={styles.fix} aria-labelledby="fix-title">
      <header className={styles.fixHead}>
        <h3 id="fix-title">
          Frame {frame}: who is who?
          <span>{clock(frame, fps)}</span>
        </h3>
        <button type="button" className={styles.close} onClick={onClose} aria-label="Close">
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </header>

      {blockers.length > 0 && (
        <ul className={styles.fixBlockers}>
          {blockers.map((e) => (
            <li key={e.id}>{eventSentence(e, label, fps)}</li>
          ))}
        </ul>
      )}

      {repair ? (
        <div className={styles.fixBody}>
          <p>
            <strong>Settled by {repair.by}.</strong> They marked {repair.anchors.length} animals on this frame; SAM 3.1
            re-tracked forward from here and every integrity check ran again. The marks are shown on the video.
          </p>
          <ul className={styles.anchorList}>
            {repair.anchors.map((a) => (
              <li key={a.subject}>
                <span className={styles.chip} style={{ background: subjectColor(a.subject, "video") }}>
                  {subjectTag(a.subject, label)}
                </span>
                <span className="mono">{a.point ? `(${a.point[0]}, ${a.point[1]})` : a.box ? `box ${a.box.join(", ")}` : ""}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : marking ? (
        <div className={styles.fixBody}>
          <p>
            Each animal starts at its current track. Choose a subject, then click its animal on the video. Swap the ones
            that are wrong; every subject needs a mark.
          </p>
          <div className={styles.markList} role="group" aria-label="Subjects to mark">
            {subjects.map((s) => {
              const done = marking.marks.some((m) => m.subject === s);
              return (
                <button key={s} type="button" aria-pressed={marking.active === s} onClick={() => live?.onActive(s)} className={styles.markButton}>
                  <span className={styles.chip} style={{ background: subjectColor(s, "video") }}>
                    {subjectTag(s, label)}
                  </span>
                  {done ? "marked" : "not marked"}
                </button>
              );
            })}
          </div>
          <label className={styles.field}>
            <span>Your name, recorded with the repair</span>
            <input value={marking.by} onChange={(e) => live?.onBy(e.target.value)} autoComplete="name" placeholder="A. Researcher" />
          </label>
          <div className={styles.fixActions}>
            <button
              type="button"
              className={styles.primary}
              disabled={!allMarked || !marking.by.trim() || marking.busy || !live?.canWrite}
              onClick={() => live?.onSubmit()}
            >
              {marking.busy ? "Re-tracking…" : `Re-track from frame ${frame}`}
            </button>
            <button type="button" className={styles.secondary} onClick={() => live?.onCancel()} disabled={marking.busy}>
              Cancel
            </button>
          </div>
          {!live?.canWrite && <p className={styles.fixNote}>Enter the operator token above to send a repair.</p>}
          {marking.message && (
            <p className={styles.fixNote} role="status">
              {marking.message}
            </p>
          )}
        </div>
      ) : live ? (
        <div className={styles.fixBody}>
          <p>
            Mark each animal on this frame. SAM 3.1 re-tracks every subject forward from here, every integrity check
            runs again, and the repair is recorded with your name.
          </p>
          <div className={styles.fixActions}>
            <button type="button" className={styles.primary} onClick={live.onStart}>
              Mark who is who
            </button>
          </div>
        </div>
      ) : (
        <div className={styles.fixBody}>
          <p>
            No repair is recorded for this run. In a live workspace a person marks each animal on this frame; SAM 3.1
            re-tracks forward from here and every integrity check runs again. From the command line, with the current
            tracks as a starting point:
          </p>
          <div className={styles.command}>
            <pre tabIndex={0} aria-label="Repair command">
              <code>{command}</code>
            </pre>
            <CopyButton text={command} tone="dark" />
          </div>
        </div>
      )}
    </section>
  );
}
