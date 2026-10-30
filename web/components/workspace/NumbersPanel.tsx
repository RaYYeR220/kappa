"use client";

import { useState } from "react";

import {
  binLabel,
  clock,
  eventSentence,
  formatValue,
  lockLine,
  metricHow,
  metricName,
  plusMinus,
  uncertaintyLines,
  unitLabel,
} from "@/lib/format";
import { subjectColor, subjectTag } from "@/lib/palette";
import type { EventView, Protocol, ResultView, Status } from "@/lib/types";

import { type MetricGroup, resultKey } from "./model";
import styles from "./Workspace.module.css";

interface Props {
  groups: MetricGroup[];
  protocol: Protocol | null;
  label: string;
  fps: number;
  events: Map<string, EventView>;
  summary: Record<Status, number>;
  selected: string | null;
  onSelect: (r: ResultView | null) => void;
  onGoTo: (frame: number, r: ResultView) => void;
  onEvidence: (frame: number) => void;
}

function LockGlyph() {
  return (
    <svg width="11" height="13" viewBox="0 0 11 13" aria-hidden="true" className={styles.lockGlyph}>
      <path d="M2.6 5.6V4a2.9 2.9 0 0 1 5.8 0v1.6" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <rect x="0.8" y="5.6" width="9.4" height="6.6" rx="1.6" fill="currentColor" />
    </svg>
  );
}

function Tick() {
  return (
    <svg width="10" height="10" viewBox="0 0 12 12" aria-hidden="true">
      <path d="M2 6.3 4.8 9 10 3.4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Cell({ r, selected, onClick, describedBy }: { r: ResultView; selected: boolean; onClick: () => void; describedBy?: string }) {
  if (r.status === "verified" && r.value != null) {
    const pm = plusMinus(r);
    return (
      <button type="button" className={styles.cell} data-status="verified" aria-pressed={selected} onClick={onClick} aria-describedby={describedBy}>
        <span className={styles.value}>
          {formatValue(r.value, r.unit)}
          <small>{unitLabel(r.unit)}</small>
        </span>
        {pm != null && <span className={styles.pm}>± {formatValue(pm, r.unit)}</span>}
        <span className="sr-only">, verified. Show its evidence.</span>
      </button>
    );
  }
  if (r.status === "locked")
    return (
      <button type="button" className={styles.cell} data-status="locked" aria-expanded={selected} onClick={onClick} aria-describedby={describedBy}>
        <span className={styles.lockLabel}>
          <LockGlyph />
          Locked
        </span>
      </button>
    );
  return (
    <span className={styles.cell} data-status="refused">
      <span className={styles.refusedLabel}>Refused</span>
    </span>
  );
}

function Detail({ r, props }: { r: ResultView; props: Props }) {
  const { events, label, fps } = props;
  if (r.status === "verified")
    return (
      <div className={styles.detail}>
        <p className={styles.detailLead}>
          <span className={styles.badgeVerified}>
            <Tick /> Verified
          </span>
          Computed from these frames. Select one to play it.
        </p>
        <ul className={styles.evidence}>
          {r.evidence.map((ev, i) => (
            <li key={i}>
              <button type="button" onClick={() => props.onEvidence(ev.span[0])}>
                <span className={styles.evSpan}>
                  frames {ev.span[0]}–{ev.span[1]}
                </span>
                <span className={styles.evNote}>{ev.note}</span>
                <span className={styles.evTime}>{clock(ev.span[0], fps)}</span>
              </button>
            </li>
          ))}
        </ul>
        {Object.keys(r.uncertainty).length > 0 && (
          <ul className={styles.notes}>
            {uncertaintyLines(r).map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        )}
      </div>
    );
  if (r.status === "locked") {
    const blockers = r.blocked_by.map((id) => events.get(id)).filter((e): e is EventView => !!e);
    const fix = r.fix_frame;
    return (
      <div className={styles.detail} data-status="locked">
        <p className={styles.detailLead}>
          Withheld: the track under this number failed {blockers.length === 1 ? "a check" : `${blockers.length} checks`}. It
          unlocks when a person marks who is who on one frame and every check passes again.
        </p>
        <ol className={styles.blockers}>
          {blockers.map((e) => (
            <li key={e.id}>
              <span className={styles.blockerFrames}>
                frames {e.span[0]}–{e.span[1]}
              </span>
              {eventSentence(e, label, fps)}
            </li>
          ))}
          {(blockers.length ? [] : r.reasons.slice(0, 3)).map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ol>
        {fix != null && (
          <button type="button" className={styles.goto} onClick={() => props.onGoTo(fix, r)}>
            Go to frame {fix}
            <span aria-hidden="true" className={styles.gotoTime}>
              {clock(fix, fps)}
            </span>
          </button>
        )}
      </div>
    );
  }
  return null;
}

export function NumbersPanel(props: Props) {
  const { groups, protocol, label, fps, events, summary, selected, onSelect } = props;
  const [open, setOpen] = useState<Set<string>>(() => new Set());

  const toggleBins = (key: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const row = (r: ResultView, title: string, sub: boolean) => {
    const key = resultKey(r);
    const isSel = selected === key;
    const reasonId = `reason-${key.replace(/[^a-z0-9]/gi, "-")}`;
    const reason =
      r.status === "locked" ? lockLine(r, events, label, fps) : r.status === "refused" ? (r.reasons[0] ?? "Not measurable from this video.") : null;
    return (
      <div className={styles.rowMain} data-sub={sub ? "" : undefined}>
        <div className={styles.rowText}>
          <span className={styles.rowTitle}>{title}</span>
          {reason && (!sub || r.status === "refused") && (
            <span id={reasonId} className={styles.rowReason}>
              {reason}
            </span>
          )}
        </div>
        <Cell r={r} selected={isSel} onClick={() => onSelect(isSel ? null : r)} describedBy={reason ? reasonId : undefined} />
      </div>
    );
  };

  return (
    <div className={styles.numbers}>
      <header className={styles.numbersHead}>
        <h2>Numbers</h2>
        <ul className={styles.counts} aria-label="Result counts">
          <li data-status="verified">
            <span aria-hidden="true">
              <Tick />
            </span>
            {summary.verified} verified
          </li>
          <li data-status="locked">
            <span aria-hidden="true">
              <LockGlyph />
            </span>
            {summary.locked} locked
          </li>
          <li data-status="refused">
            <span aria-hidden="true" />
            {summary.refused} refused
          </li>
        </ul>
      </header>
      {groups.map((g) => (
        <section key={g.id} className={styles.metric} aria-labelledby={`m-${g.id}`}>
          <header className={styles.metricHead}>
            <h3 id={`m-${g.id}`}>{metricName(g.spec, protocol, g.id)}</h3>
            <p>{metricHow(g.spec, protocol)}</p>
          </header>
          <ul className={styles.rows}>
            {g.rows.map((rowData) => {
              const head = rowData.whole ?? rowData.bins[0];
              if (!head) return null;
              const binKey = `${g.id}|${rowData.subject ?? "all"}`;
              const expanded = open.has(binKey);
              const selectedHere = [rowData.whole, ...rowData.bins].find((r) => r && resultKey(r) === selected);
              return (
                <li key={binKey} className={styles.row}>
                  <div className={styles.rowLine}>
                    <span
                      className={styles.chip}
                      style={{ background: rowData.subject != null ? subjectColor(rowData.subject, "video") : "var(--g3)" }}
                    >
                      {subjectTag(rowData.subject, label)}
                    </span>
                    {rowData.whole ? row(rowData.whole, "Whole session", false) : row(rowData.bins[0]!, binLabel(rowData.bins[0]!), true)}
                  </div>
                  {selectedHere && selectedHere === rowData.whole && <Detail r={selectedHere} props={props} />}
                  {rowData.bins.length > 0 && rowData.whole && (
                    <>
                      <button
                        type="button"
                        className={styles.binsToggle}
                        aria-expanded={expanded}
                        aria-controls={`bins-${binKey}`}
                        onClick={() => toggleBins(binKey)}
                      >
                        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
                          <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        {expanded ? "Hide" : "Show"} {rowData.bins.length} bins
                      </button>
                      <ul id={`bins-${binKey}`} className={styles.bins} hidden={!expanded}>
                        {rowData.bins.map((b) => (
                          <li key={resultKey(b)}>
                            {row(b, binLabel(b), true)}
                            {selectedHere === b && <Detail r={b} props={props} />}
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      ))}
      {protocol?.refused.length ? (
        <section className={styles.metric} aria-labelledby="m-refused">
          <header className={styles.metricHead}>
            <h3 id="m-refused">Refused by the protocol</h3>
            <p>Asked for, and not measurable from this camera.</p>
          </header>
          <ul className={styles.rows}>
            {protocol.refused.map((r) => (
              <li key={r.request} className={styles.row}>
                <div className={styles.rowLine}>
                  <div className={styles.rowMain}>
                    <div className={styles.rowText}>
                      <span className={styles.rowTitle}>{r.request}</span>
                      <span className={styles.rowReason}>{r.reason}</span>
                    </div>
                    <span className={styles.cell} data-status="refused">
                      <span className={styles.refusedLabel}>Refused</span>
                    </span>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
