"use client";

import { memo, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { clock, clockShort, eventSentence, resolvedBy } from "@/lib/format";
import { subjectColor, subjectTag } from "@/lib/palette";
import type { Player } from "@/lib/player";
import type { EventView, Span, TracksFile } from "@/lib/types";

import { isOpenBlocker, lockedFrom } from "./model";
import styles from "./Workspace.module.css";

export interface TimelineFocus {
  /** Evidence of the selected verified number. */
  evidence: { span: Span; subject: number | null }[];
  /** Events locking the selected number. */
  blockers: Set<string>;
  /** The selected number's time bin. */
  window: Span | null;
}

const LABEL_W = 46;
const AXIS_H = 26;
const LANE_H = 22;
const GAP = 5;
const PAD_R = 6;

interface Props {
  player: Player;
  subjects: number[];
  label: string;
  tracks: TracksFile | null;
  events: EventView[];
  bins: number[];
  focus: TimelineFocus | null;
}

function runs(values: boolean[]): Span[] {
  const out: Span[] = [];
  let start = -1;
  for (let i = 0; i <= values.length; i++) {
    const on = i < values.length && values[i];
    if (on && start < 0) start = i;
    if (!on && start >= 0) {
      out.push([start, i - 1]);
      start = -1;
    }
  }
  return out;
}

function tickStep(seconds: number, width: number): number {
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  for (const s of steps) if ((width / seconds) * s >= 56) return s;
  return 900;
}

const Lanes = memo(function Lanes({
  width,
  n,
  fps,
  subjects,
  label,
  tracks,
  events,
  bins,
  focus,
  allLane,
}: {
  width: number;
  n: number;
  fps: number;
  subjects: number[];
  label: string;
  tracks: TracksFile | null;
  events: EventView[];
  bins: number[];
  focus: TimelineFocus | null;
  allLane: boolean;
}) {
  const lanes: (number | null)[] = [...subjects, ...(allLane ? [null] : [])];
  const height = AXIS_H + lanes.length * (LANE_H + GAP) + 4;
  const span = Math.max(1, width - LABEL_W - PAD_R);
  const X = (f: number) => LABEL_W + (Math.max(0, Math.min(n, f)) / n) * span;
  const laneY = (i: number) => AXIS_H + i * (LANE_H + GAP);
  const laneOf = (s: number | null) => lanes.indexOf(s);
  const locked = lockedFrom(events);
  const seconds = n / fps;
  const step = tickStep(seconds, span);
  const ticks: number[] = [];
  for (let s = 0; s <= seconds + 1e-6; s += step) ticks.push(s);
  const dimOthers = !!focus && (focus.blockers.size > 0 || focus.evidence.length > 0);

  return (
    <svg width={width} height={height} className={styles.lanes} aria-hidden="true">
      <defs>
        <pattern id="tl-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill="#e9e7e2" />
          <line x1="0" y1="0" x2="0" y2="6" stroke="#0e0e0c" strokeOpacity=".55" strokeWidth="1.2" />
        </pattern>
        <pattern id="tl-hatch-soft" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="#0e0e0c" strokeOpacity=".16" strokeWidth="1" />
        </pattern>
      </defs>

      {/* time axis */}
      {ticks.map((s) => (
        <g key={s}>
          <line x1={X(s * fps)} x2={X(s * fps)} y1={AXIS_H - 6} y2={height - 2} stroke="#0e0e0c" strokeOpacity=".07" />
          <text x={X(s * fps)} y={12} className={styles.tick} textAnchor={s === 0 ? "start" : "middle"}>
            {clockShort(s)}
          </text>
        </g>
      ))}
      {bins.map((f) => (
        <line key={`b${f}`} x1={X(f)} x2={X(f)} y1={AXIS_H - 2} y2={height - 2} stroke="#8b8983" strokeDasharray="2 3" />
      ))}
      {focus?.window && (
        <g>
          <rect x={X(focus.window[0])} y={AXIS_H - 5} width={X(focus.window[1]) - X(focus.window[0])} height={2.5} rx={1} fill="#2448e8" />
        </g>
      )}

      {lanes.map((s, i) => {
        const y = laneY(i);
        const t = s != null ? tracks?.subjects[String(s)] : undefined;
        const present = t ? runs(t.cx.map((v) => v != null)) : [];
        const interp = t ? runs(t.interpolated.map((v) => v === 1)) : [];
        const lockStart = s != null ? locked.get(s) : undefined;
        return (
          <g key={s ?? "all"}>
            <rect x={LABEL_W} y={y} width={span} height={LANE_H} rx={4} fill="#0e0e0c" fillOpacity=".035" />
            {lockStart !== undefined && (
              <rect x={X(lockStart)} y={y} width={X(n) - X(lockStart)} height={LANE_H} rx={4} fill="url(#tl-hatch-soft)" />
            )}
            {present.map(([a, b]) => (
              <line
                key={`p${a}`}
                x1={X(a)}
                x2={X(b + 1)}
                y1={y + LANE_H / 2}
                y2={y + LANE_H / 2}
                stroke={s != null ? subjectColor(s, "light") : "#565550"}
                strokeWidth={2}
                strokeLinecap="round"
              />
            ))}
            {interp.map(([a, b]) => (
              <line key={`i${a}`} x1={X(a)} x2={X(b + 1)} y1={y + LANE_H / 2} y2={y + LANE_H / 2} stroke="#f1f0ec" strokeWidth={2} strokeDasharray="2 2" />
            ))}
            <g transform={`translate(0 ${y + 2})`}>
              <rect x={2} width={34} height={LANE_H - 4} rx={5} fill={s != null ? subjectColor(s, "video") : "#dedcd5"} />
              <text x={19} y={(LANE_H - 4) / 2 + 4} textAnchor="middle" className={styles.laneTag}>
                {subjectTag(s, label)}
              </text>
            </g>
          </g>
        );
      })}

      {events.map((e) => {
        const targets = e.subjects.length ? e.subjects.map(laneOf).filter((i) => i >= 0) : [laneOf(null)].filter((i) => i >= 0);
        const x0 = X(e.span[0]);
        const w = Math.max(2.5, X(e.span[1] + 1) - x0);
        const open = isOpenBlocker(e);
        const person = !!e.resolved_by && e.resolved_by !== "motion-continuity";
        const emphasised = focus?.blockers.has(e.id);
        const opacity = dimOthers && !emphasised ? 0.3 : 1;
        const title = `${eventSentence(e, label, fps)}${e.resolved_by ? ` (${resolvedBy(e)})` : ""}`;
        return (
          <g key={e.id} opacity={opacity}>
            <title>{title}</title>
            {targets.map((i) => {
              const y = laneY(i);
              if (open)
                return (
                  <rect
                    key={i}
                    x={x0}
                    y={y + 1}
                    width={w}
                    height={LANE_H - 2}
                    rx={3}
                    fill="url(#tl-hatch)"
                    stroke={emphasised ? "#2448e8" : "#0e0e0c"}
                    strokeOpacity={emphasised ? 1 : 0.45}
                    strokeWidth={emphasised ? 1.6 : 1}
                  />
                );
              return (
                <g key={i}>
                  <rect
                    x={x0}
                    y={y + 4}
                    width={w}
                    height={LANE_H - 8}
                    rx={3}
                    fill={person ? "#2448e8" : "#8b8983"}
                    fillOpacity={person ? 0.22 : 0.3}
                    stroke={person ? "#2448e8" : "none"}
                    strokeOpacity={0.6}
                  />
                  {w >= 12 && (
                    <path
                      d={`M${x0 + w / 2 - 3.5} ${y + LANE_H / 2}l2.4 2.4 4.6-5`}
                      fill="none"
                      stroke={person ? "#1a35c4" : "#0e0e0c"}
                      strokeWidth={1.4}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  )}
                </g>
              );
            })}
            {emphasised &&
              e.fix_frame != null &&
              targets.map((i) => (
                <path key={`f${i}`} d={`M${X(e.fix_frame! + 0.5)} ${laneY(i) - 3}l3.5 -4.5h-7z`} fill="#2448e8" />
              ))}
          </g>
        );
      })}

      {focus?.evidence.map((ev, k) => {
        const rows = ev.subject == null ? lanes.map((_, i) => i) : [laneOf(ev.subject)].filter((i) => i >= 0);
        if (!rows.length) return null;
        const top = laneY(Math.min(...rows)) - 2;
        const bottom = laneY(Math.max(...rows)) + LANE_H + 2;
        const x0 = X(ev.span[0]);
        const w = Math.max(3, X(ev.span[1]) - x0);
        return (
          <g key={k}>
            <rect x={x0} y={top} width={w} height={bottom - top} rx={4} fill="#2448e8" fillOpacity=".14" stroke="#2448e8" strokeWidth={1.5} />
          </g>
        );
      })}
    </svg>
  );
});

function Playhead({ player, width, height }: { player: Player; width: number; height: number }) {
  const frame = useSyncExternalStore(player.subscribe, player.getFrame, player.getFrame);
  const span = Math.max(1, width - LABEL_W - PAD_R);
  const x = LABEL_W + (frame / player.nFrames) * span;
  return (
    <div className={styles.playhead} style={{ transform: `translateX(${x}px)`, height }} aria-hidden="true">
      <span />
    </div>
  );
}

export function Timeline({ player, subjects, label, tracks, events, bins, focus }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const frame = useSyncExternalStore(player.subscribe, player.getFrame, player.getFrame);
  const dragging = useRef(false);
  const pending = useRef<number | null>(null);
  const allLane = useMemo(() => events.some((e) => e.subjects.length === 0), [events]);
  const laneCount = subjects.length + (allLane ? 1 : 0);
  const height = AXIS_H + laneCount * (LANE_H + GAP) + 4;

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry?.contentRect.width ?? 0)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const frameAtX = (clientX: number) => {
    const r = wrap.current!.getBoundingClientRect();
    const span = Math.max(1, r.width - LABEL_W - PAD_R);
    return Math.round(((clientX - r.left - LABEL_W) / span) * player.nFrames);
  };

  const seekTo = (f: number) => {
    pending.current = f;
    requestAnimationFrame(() => {
      if (pending.current != null) {
        player.seek(pending.current);
        pending.current = null;
      }
    });
  };

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return;
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    player.pause();
    seekTo(frameAtX(e.clientX));
  }

  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (dragging.current) seekTo(frameAtX(e.clientX));
  }

  function onPointerUp(e: React.PointerEvent<HTMLDivElement>) {
    dragging.current = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    const second = Math.round(player.fps);
    let next: number | null = null;
    if (e.key === "ArrowRight") next = frame + (e.shiftKey ? second : 1);
    else if (e.key === "ArrowLeft") next = frame - (e.shiftKey ? second : 1);
    else if (e.key === "PageUp") next = frame + second * 5;
    else if (e.key === "PageDown") next = frame - second * 5;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = player.nFrames - 1;
    if (next == null) return;
    e.preventDefault();
    player.pause();
    player.seek(next);
  }

  return (
    <div
      ref={wrap}
      className={styles.timeline}
      role="slider"
      tabIndex={0}
      aria-label="Timeline. Arrow keys step one frame, Shift with an arrow steps one second."
      aria-valuemin={0}
      aria-valuemax={player.nFrames - 1}
      aria-valuenow={frame}
      aria-valuetext={`Frame ${frame}, ${clock(frame, player.fps)}`}
      style={{ height }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onKeyDown={onKeyDown}
    >
      {width > 0 && (
        <>
          <Lanes
            width={width}
            n={player.nFrames}
            fps={player.fps}
            subjects={subjects}
            label={label}
            tracks={tracks}
            events={events}
            bins={bins}
            focus={focus}
            allLane={allLane}
          />
          <Playhead player={player} width={width} height={height} />
        </>
      )}
    </div>
  );
}

export function TimelineLegend() {
  return (
    <ul className={styles.legend} aria-label="Timeline key">
      <li>
        <span className={styles.keyOpen} aria-hidden="true" /> Open integrity check, numbers after it locked
      </li>
      <li>
        <span className={styles.keyMinor} aria-hidden="true" /> Settled by the motion
      </li>
      <li>
        <span className={styles.keyPerson} aria-hidden="true" /> Settled by a person
      </li>
      <li>
        <span className={styles.keyEvidence} aria-hidden="true" /> Evidence of the selected number
      </li>
    </ul>
  );
}

