"use client";

import { useEffect, useRef, useState } from "react";

import styles from "./RefusalDemo.module.css";

type Stage = "locked" | "marking" | "retracking" | "verified";

const SEGMENTS = 12;
const FIX = 7; // the fix frame sits at segment 7 of 12

// a mouse seen from above, nose to the right, in its own coordinates
const BODY =
  "M-25 0C-25-11.5-9-14.5 5-11.5C13-10 19.5-6.5 25.5-2.4C28.4-.6 28.4.6 25.5 2.4C19.5 6.5 13 10 5 11.5C-9 14.5-25 11.5-25 0Z";
const EARS = "M9.5-8.6a4.2 4.2 0 1 0 .01 0ZM9.5 8.6a4.2 4.2 0 1 0 .01 0Z";
const TAIL = "M-25 0C-38 2-44-9-58-6";

function Mouse({
  x,
  y,
  angle,
  hue,
  line,
  uncertain,
}: {
  x: number;
  y: number;
  angle: number;
  hue: string;
  line: string;
  uncertain: boolean;
}) {
  return (
    <g transform={`translate(${x} ${y}) rotate(${angle})`} className={styles.mouse}>
      <path d={TAIL} fill="none" stroke={uncertain ? "#8b8983" : line} strokeWidth="2" strokeLinecap="round" />
      <path d={`${BODY}${EARS}`} fill="none" stroke="#0b0b0a" strokeOpacity={uncertain ? 0 : 0.7} strokeWidth="4.4" />
      <path
        d={`${BODY}${EARS}`}
        fill={uncertain ? "#c5c3bb" : hue}
        fillOpacity={uncertain ? 0.55 : 0.5}
        stroke={uncertain ? "#565550" : line}
        strokeWidth="2.2"
        strokeDasharray={uncertain ? "4 3" : undefined}
      />
    </g>
  );
}

export function RefusalDemo() {
  const [stage, setStage] = useState<Stage>("locked");
  const [picked, setPicked] = useState<"a" | "b" | null>(null);
  const timers = useRef<number[]>([]);

  useEffect(() => () => timers.current.forEach((t) => window.clearTimeout(t)), []);

  const reduced = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function pick(which: "a" | "b") {
    if (stage !== "marking") return;
    setPicked(which);
    setStage("retracking");
    const wait = reduced() ? 50 : 1500;
    timers.current.push(window.setTimeout(() => setStage("verified"), wait));
  }

  function reset() {
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current = [];
    setPicked(null);
    setStage("locked");
  }

  const uncertain = stage === "locked" || stage === "marking";
  // after the mark, the animal the person picked is M2 (coral); the other is M3 (blue)
  const aIsM2 = picked !== "b";
  const tagA = uncertain ? "?" : aIsM2 ? "M2" : "M3";
  const tagB = uncertain ? "?" : aIsM2 ? "M3" : "M2";
  const coral = { hue: "#f9786a", line: "#bc4a3f" };
  const blue = { hue: "#52a9fe", line: "#1577c8" };
  const styleA = aIsM2 ? coral : blue;
  const styleB = aIsM2 ? blue : coral;

  const status =
    stage === "locked"
      ? "Distance for M2 is locked: M2 and M3 touched at 03:12 and their motion afterwards fits either identity."
      : stage === "marking"
        ? "Marking frame 4,680. Select the animal that is M2."
        : stage === "retracking"
          ? "SAM 3.1 is re-tracking forward from frame 4,680. Every integrity check runs again."
          : "Verified: M2 travelled 41.2 metres. Identity settled by you on frame 4,680.";

  return (
    <div className={styles.demo} data-stage={stage}>
      <div className={styles.well}>
        <svg viewBox="0 0 480 270" className={styles.frame} role="img" aria-label="Two mice in an open-field arena, just after they touched">
          <rect x="0" y="0" width="480" height="270" fill="#d8d5cc" />
          <rect x="70" y="18" width="340" height="234" rx="6" fill="#e4e1d9" stroke="#bdb9ae" />
          <rect x="155" y="76" width="170" height="117" fill="none" stroke="#bdb9ae" strokeDasharray="3 4" />
          <Mouse x={214} y={150} angle={-18} {...styleA} uncertain={uncertain} />
          <Mouse x={268} y={128} angle={164} {...styleB} uncertain={uncertain} />
          <g className={styles.tags} fontFamily="var(--mono)" fontSize="11" fontWeight="500">
            <g transform="translate(170 178)">
              <rect width="30" height="19" rx="5" fill={uncertain ? "#e9e7e2" : styleA.hue} stroke={uncertain ? "#565550" : "none"} />
              <text x="15" y="13.4" textAnchor="middle" fill="#0e0e0c">
                {tagA}
              </text>
            </g>
            <g transform="translate(286 90)">
              <rect width="30" height="19" rx="5" fill={uncertain ? "#e9e7e2" : styleB.hue} stroke={uncertain ? "#565550" : "none"} />
              <text x="15" y="13.4" textAnchor="middle" fill="#0e0e0c">
                {tagB}
              </text>
            </g>
          </g>
        </svg>
        {stage === "marking" && (
          <div className={styles.targets}>
            <button type="button" className={styles.target} style={{ left: "44.6%", top: "55.6%" }} onClick={() => pick("a")} aria-label="This one is M2 (lower left animal)" />
            <button type="button" className={styles.target} style={{ left: "55.8%", top: "47.4%" }} onClick={() => pick("b")} aria-label="This one is M2 (upper right animal)" />
            <p className={styles.prompt}>Select the animal that is M2</p>
          </div>
        )}
      </div>

      <div className={styles.timeline} aria-hidden="true">
        <span className={styles.laneLabel}>M2</span>
        <div className={styles.lane}>
          {Array.from({ length: SEGMENTS }, (_, i) => (
            <span
              key={i}
              className={`${styles.seg} ${i >= FIX ? styles.after : ""}`}
              style={{ "--k": Math.max(0, i - FIX) } as React.CSSProperties}
            />
          ))}
          <span className={styles.contact} style={{ left: `${((FIX - 0.6) / SEGMENTS) * 100}%` }} />
          <span className={styles.fix} style={{ left: `${(FIX / SEGMENTS) * 100}%` }}>
            <span>frame 4,680</span>
          </span>
        </div>
      </div>

      <div className={styles.row}>
        <div className={styles.metric}>
          <span>Distance travelled</span>
          <span className={styles.who}>M2, whole session</span>
        </div>
        <div className={styles.cell} data-state={stage === "verified" ? "verified" : "locked"}>
          {stage === "verified" ? (
            <span className={styles.value}>
              <span className={styles.digits}>41.2</span> m
              <span className={styles.check} aria-hidden="true">
                <svg width="12" height="12" viewBox="0 0 12 12">
                  <path d="M2 6.3 4.8 9 10 3.4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
            </span>
          ) : (
            <span className={styles.lockedText}>
              <svg width="11" height="13" viewBox="0 0 11 13" aria-hidden="true">
                <path d="M2.6 5.6V4a2.9 2.9 0 0 1 5.8 0v1.6" fill="none" stroke="currentColor" strokeWidth="1.3" />
                <rect x="0.8" y="5.6" width="9.4" height="6.6" rx="1.6" fill="currentColor" />
              </svg>
              {stage === "retracking" ? "Checking again" : "Locked"}
            </span>
          )}
        </div>
      </div>

      <p className={styles.status} aria-live="polite">
        {status}
      </p>

      <div className={styles.actions}>
        {stage === "locked" && (
          <button type="button" className={styles.primary} onClick={() => setStage("marking")}>
            Mark who is who on frame 4,680
          </button>
        )}
        {stage === "marking" && (
          <button type="button" className={styles.secondary} onClick={reset}>
            Cancel
          </button>
        )}
        {stage === "verified" && (
          <button type="button" className={styles.secondary} onClick={reset}>
            Play it again
          </button>
        )}
      </div>
    </div>
  );
}
