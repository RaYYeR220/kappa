"use client";

import { useState } from "react";

import { Code } from "@/components/ui/Code";
import { CopyButton } from "@/components/ui/CopyButton";
import { clock, eventSentence, eventTitle, metricName, modelName, plural, resolvedBy } from "@/lib/format";
import { subjectColor, subjectTag } from "@/lib/palette";
import type { EventView, Review, RunData, Zone } from "@/lib/types";

import { isOpenBlocker, toCsv } from "./model";
import styles from "./Workspace.module.css";

function zoneText(z: Zone): string {
  if (z.kind === "periphery_band") return `within ${z.band_cm} cm of the wall`;
  if (z.kind === "center_fraction") return `the central ${Math.round((z.fraction ?? 0) * 100)}% of the floor`;
  if (z.kind === "rect_cm" && z.rect_cm) return `rectangle ${z.rect_cm.join(", ")} cm`;
  return z.kind;
}

export function ProtocolCard({ run }: { run: RunData }) {
  const p = run.protocol;
  if (!p) {
    return (
      <section className={styles.card} aria-labelledby="protocol-title">
        <h2 id="protocol-title">Protocol</h2>
        <p className={styles.muted}>This run has no confirmed protocol yet. Nothing is measured until a person confirms one.</p>
      </section>
    );
  }
  const integrity = p.integrity;
  return (
    <section className={styles.card} aria-labelledby="protocol-title">
      <h2 id="protocol-title">Protocol</h2>
      <blockquote className={styles.question}>{p.question}</blockquote>
      <dl className={styles.facts}>
        <div>
          <dt>Subjects</dt>
          <dd>
            {p.subject.expected_count} {p.subject.expected_count === 1 ? p.subject.label : plural(p.subject.label)}, prompted to SAM 3.1 as{" "}
            <q>{p.subject.concept}</q>
          </dd>
        </div>
        <div>
          <dt>Arena</dt>
          <dd>
            {p.arena.shape === "rect" ? "Rectangle" : "Circle"}, {p.arena.width_cm} × {p.arena.height_cm} cm,{" "}
            {p.arena.corners_px ? "floor corners marked on the video" : "corners not marked yet"}
          </dd>
        </div>
        {p.zones.length > 0 && (
          <div>
            <dt>Zones</dt>
            <dd>
              <ul className={styles.inline}>
                {p.zones.map((z) => (
                  <li key={z.id}>
                    <code>{z.id}</code> {zoneText(z)}
                  </li>
                ))}
              </ul>
            </dd>
          </div>
        )}
        {p.entry_rule && (
          <div>
            <dt>Inside a zone</dt>
            <dd>
              {p.entry_rule.mode === "centroid"
                ? "when the body centre is inside"
                : `when at least ${Math.round(p.entry_rule.body_fraction * 100)}% of the body is inside`}
              {p.entry_rule.min_dwell_s > 0 ? `, for at least ${p.entry_rule.min_dwell_s} s` : ""}
            </dd>
          </div>
        )}
        <div>
          <dt>Metrics</dt>
          <dd>
            <ul className={styles.inline}>
              {p.metrics.map((m) => (
                <li key={m.id}>
                  <code>{m.id}</code> {metricName(m, p, m.id).toLowerCase()}
                  {m.bin_s ? `, per ${m.bin_s} s` : ""}
                </li>
              ))}
            </ul>
          </dd>
        </div>
        {p.behaviours.length > 0 && (
          <div>
            <dt>Behaviours</dt>
            <dd>
              <ul className={styles.inline}>
                {p.behaviours.map((b) => (
                  <li key={b.id}>
                    <strong>{b.label}.</strong> {b.definition}
                  </li>
                ))}
              </ul>
            </dd>
          </div>
        )}
        {integrity && (
          <div>
            <dt>Integrity checks</dt>
            <dd>
              Contact when centres come within {integrity.contact_distance} body lengths; identities kept only when keeping
              them fits the motion {integrity.identity_margin}× better than swapping. Faster than {integrity.max_speed_cm_s}{" "}
              cm/s is a jump. Gaps up to {integrity.max_interpolated_gap_s} s are filled. Confidence below{" "}
              {integrity.min_confidence} for {integrity.low_confidence_min_s} s is flagged.
            </dd>
          </div>
        )}
        {p.smoothing && (
          <div>
            <dt>Smoothing</dt>
            <dd>
              {p.smoothing.method === "none" ? "none" : `${p.smoothing.method} filter, ${p.smoothing.window_s} s window`}
              {p.smoothing.min_step_cm > 0 ? `; steps under ${p.smoothing.min_step_cm} cm ignored` : ""}
            </dd>
          </div>
        )}
        {p.refused.length > 0 && (
          <div>
            <dt>Refused</dt>
            <dd>
              <ul className={styles.inline}>
                {p.refused.map((r) => (
                  <li key={r.request}>
                    <strong>{r.request}.</strong> {r.reason}.
                  </li>
                ))}
              </ul>
            </dd>
          </div>
        )}
        {(run.assumptions.length > 0 || p.notes) && (
          <div>
            <dt>Assumptions</dt>
            <dd>
              {run.assumptions.length > 0 && (
                <ul className={styles.inline}>
                  {run.assumptions.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              )}
              {p.notes && <p className={styles.notesText}>{p.notes}</p>}
            </dd>
          </div>
        )}
        <div>
          <dt>Confirmed by</dt>
          <dd>{run.confirmedBy ?? "not confirmed"}</dd>
        </div>
        {run.protocolSha256 && (
          <div>
            <dt>sha256</dt>
            <dd className={styles.sha}>
              <code>{run.protocolSha256}</code>
              <CopyButton text={run.protocolSha256} />
            </dd>
          </div>
        )}
      </dl>
      <details className={styles.json}>
        <summary>Protocol JSON</summary>
        <Code code={JSON.stringify(p, null, 2)} label="protocol.json" language="json" />
      </details>
    </section>
  );
}

const VERDICT: Record<Review["verdict"], string> = {
  no_concerns: "No concerns",
  minor_concerns: "Minor concerns",
  major_concerns: "Major concerns",
};

export function ReviewCard({
  run,
  onRef,
  live,
}: {
  run: RunData;
  onRef: (id: string) => void;
  live: { canWrite: boolean; busy: boolean; message: string | null; onRequest: () => void } | null;
}) {
  const review = run.manifest?.review.at(-1) ?? null;
  const ids = new Set([...run.events.map((e) => e.id), ...run.results.map((r) => r.metric_id)]);
  return (
    <section className={styles.card} aria-labelledby="review-title">
      <h2 id="review-title">Reviewer 2</h2>
      <p className={styles.muted}>
        An adversarial read of the whole run by Muse Wonder, looking for reasons a careful reviewer would not accept the
        numbers. Advice only: it never changes a number.
      </p>
      {review ? (
        <>
          <p className={styles.verdict} data-verdict={review.verdict}>
            {VERDICT[review.verdict]}
            {review.model && <span>{modelName(review.model)}</span>}
          </p>
          {review.findings.length > 0 ? (
            <ol className={styles.findings}>
              {review.findings.map((f, i) => (
                <li key={i} data-severity={f.severity}>
                  <p className={styles.findingMeta}>
                    <span className={styles.severity}>{f.severity}</span>
                    <span>{f.topic}</span>
                  </p>
                  <p className={styles.claim}>{f.claim}</p>
                  {f.suggestion && <p className={styles.suggestion}>{f.suggestion}</p>}
                  {f.refs.length > 0 && (
                    <p className={styles.refs}>
                      {f.refs.map((ref) =>
                        ids.has(ref) ? (
                          <button key={ref} type="button" onClick={() => onRef(ref)}>
                            {ref}
                          </button>
                        ) : (
                          <code key={ref}>{ref}</code>
                        ),
                      )}
                    </p>
                  )}
                </li>
              ))}
            </ol>
          ) : (
            <p className={styles.muted}>Nothing material found.</p>
          )}
        </>
      ) : (
        <p className={styles.empty}>No review on this run yet.</p>
      )}
      {live && (
        <div className={styles.cardActions}>
          <button type="button" className={styles.secondary} disabled={!live.canWrite || live.busy} onClick={live.onRequest}>
            {live.busy ? "Reviewing…" : review ? "Ask for a new review" : "Ask Reviewer 2"}
          </button>
          {live.message && (
            <p className={styles.fixNote} role="status">
              {live.message}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

export function MethodsCard({ run }: { run: RunData }) {
  const methods = run.manifest?.methods;
  return (
    <section className={styles.card} aria-labelledby="methods-title">
      <div className={styles.cardHead}>
        <h2 id="methods-title">Methods</h2>
        {methods && <CopyButton text={methods} label="Copy paragraph" />}
      </div>
      {methods ? <p className={styles.methods}>{methods}</p> : <p className={styles.empty}>Written with the manifest once the run is analysed.</p>}
    </section>
  );
}

export function IntegrityList({ run, label, onJump }: { run: RunData; label: string; onJump: (e: EventView) => void }) {
  const [showSettled, setShowSettled] = useState(false);
  const fps = run.video.fps;
  const open = run.events.filter(isOpenBlocker);
  const settled = run.events.filter((e) => !isOpenBlocker(e));
  const item = (e: EventView) => (
    <li key={e.id} data-open={isOpenBlocker(e) ? "" : undefined}>
      <button type="button" onClick={() => onJump(e)}>
        <span className={styles.eventChips} aria-hidden="true">
          {e.subjects.map((s) => (
            <span key={s} className={styles.chip} style={{ background: subjectColor(s, "video") }}>
              {subjectTag(s, label)}
            </span>
          ))}
        </span>
        <span className={styles.eventText}>
          <span className={styles.eventTitle}>{eventTitle(e, label)}</span>
          <span className={styles.eventSentence}>
            {eventSentence(e, label, fps)}
            {e.resolved_by ? ` ${resolvedBy(e)?.replace(/^./, (c) => c.toUpperCase())}.` : ""}
          </span>
        </span>
        <span className={styles.eventFrames}>
          {e.fix_frame != null && isOpenBlocker(e) ? `fix at ${e.fix_frame}` : clock(e.span[0], fps)}
        </span>
      </button>
    </li>
  );
  return (
    <section className={styles.card} aria-labelledby="integrity-title">
      <h2 id="integrity-title">Integrity checks</h2>
      <p className={styles.muted}>
        {open.length} open, {settled.length} settled. An open check locks every number that depends on it.
      </p>
      {open.length > 0 && <ul className={styles.events}>{open.map(item)}</ul>}
      {settled.length > 0 && (
        <>
          <button type="button" className={styles.binsToggle} aria-expanded={showSettled} onClick={() => setShowSettled((v) => !v)}>
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
              <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {showSettled ? "Hide" : "Show"} {settled.length} settled
          </button>
          {showSettled && <ul className={styles.events}>{settled.map(item)}</ul>}
        </>
      )}
    </section>
  );
}

export function VerdictsCard({ run, label, onJump }: { run: RunData; label: string; onJump: (frame: number) => void }) {
  const entries = Object.entries(run.verdicts).filter(([, v]) => Array.isArray(v) && v.length);
  if (!entries.length) return null;
  const fps = run.video.fps;
  return (
    <section className={`${styles.card} ${styles.wide}`} aria-labelledby="verdicts-title">
      <h2 id="verdicts-title">Behaviour bouts</h2>
      <p className={styles.muted}>
        Bouts read frame by frame from the SAM 3.1 silhouettes, each with the detector&apos;s mean probability. Only
        bouts behind verified numbers are listed.
      </p>
      {entries.map(([behaviour, bouts]) => (
        <div key={behaviour} className={styles.verdictGroup}>
          <h3>{run.protocol?.behaviours.find((b) => b.id === behaviour)?.label ?? behaviour}</h3>
          <ul className={styles.verdictList}>
            {bouts.map((v, i) => (
              <li key={i}>
                <button type="button" onClick={() => onJump(v.span[0])}>
                  <span className={styles.chip} style={{ background: subjectColor(v.subject, "video") }}>
                    {subjectTag(v.subject, label)}
                  </span>
                  <span className="mono">{clock(v.span[0], fps)}</span>
                  <span className={styles.verdictCall} data-present="true">
                    {((v.span[1] - v.span[0]) / fps).toFixed(1)} s · p {v.probability.toFixed(2)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  );
}

function save(name: string, body: string, type: string) {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function Downloads({ run }: { run: RunData }) {
  return (
    <section className={styles.card} aria-labelledby="downloads-title">
      <h2 id="downloads-title">Downloads</h2>
      <ul className={styles.downloads}>
        {run.urls.manifest && (
          <li>
            <a href={run.urls.manifest} download={`kappa-${run.id}-manifest.json`}>
              <span>Manifest</span>
              <span className={styles.dlMeta}>JSON, every result with its status, events, models and hashes</span>
            </a>
          </li>
        )}
        <li>
          <button type="button" onClick={() => save(`kappa-${run.id}-results.csv`, toCsv(run), "text/csv")}>
            <span>Results</span>
            <span className={styles.dlMeta}>CSV, one row per number; locked values stay empty</span>
          </button>
        </li>
        {run.protocol && (
          <li>
            <button type="button" onClick={() => save(`kappa-${run.id}-protocol.json`, JSON.stringify(run.protocol, null, 2), "application/json")}>
              <span>Protocol</span>
              <span className={styles.dlMeta}>JSON, the confirmed assay</span>
            </button>
          </li>
        )}
      </ul>
    </section>
  );
}
