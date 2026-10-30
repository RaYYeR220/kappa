"use client";

import { useEffect, useRef } from "react";

import { subjectColor, subjectTag } from "@/lib/palette";
import type { Player } from "@/lib/player";
import type { MaskOverlay } from "@/lib/sam/overlay";
import type { EventView, TracksFile } from "@/lib/types";

import { IDENTITY_KINDS, isOpenBlocker, lockedFrom } from "./model";
import styles from "./Workspace.module.css";

export interface Layers {
  masks: boolean;
  trails: boolean;
  tags: boolean;
}

export interface Mark {
  subject: number;
  x: number;
  y: number;
}

interface Props {
  src: string;
  width: number;
  height: number;
  tracks: TracksFile | null;
  overlay: MaskOverlay;
  version: number;
  player: Player;
  layers: Layers;
  label: string;
  events: EventView[];
  focusSubject: number | null;
  /** Points a person marked on a repair frame, shown on that frame. */
  anchors: { frame: number; marks: Mark[]; by: string } | null;
  /** Live marking: the subject being placed and the marks so far. */
  marking: { frame: number; active: number | null; marks: Mark[]; onMark: (x: number, y: number) => void } | null;
  children?: React.ReactNode;
}

const TRAIL_S = 0.8;

interface Chip {
  text: string;
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string | null;
  locked: boolean;
  uncertain: boolean;
  focus: boolean;
}

export function VideoStage(props: Props) {
  const stage = useRef<HTMLDivElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const latest = useRef(props);
  const hatch = useRef<CanvasPattern | null>(null);
  const font = useRef("ui-monospace, monospace");

  useEffect(() => {
    latest.current = props;
  });

  useEffect(() => {
    const mono = getComputedStyle(document.documentElement).getPropertyValue("--font-geist-mono").trim();
    if (mono) font.current = `${mono}, ui-monospace, monospace`;
  }, []);

  // one draw: trails under masks under tags, all in canvas backing pixels
  const draw = useRef(() => {});
  useEffect(() => {
    draw.current = () => {
      const cv = canvas.current;
      const ctx = cv?.getContext("2d");
      if (!cv || !ctx) return;
      const p = latest.current;
      const frame = p.player.getFrame();
      const dpr = cv.width / Math.max(1, cv.clientWidth);
      const sx = cv.width / p.width;
      const sy = cv.height / p.height;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, cv.width, cv.height);
      const tracks = p.tracks;
      const subjects = tracks ? Object.entries(tracks.subjects).map(([k, t]) => [Number(k), t] as const) : [];

      if (p.layers.trails && tracks) {
        const span = Math.round(TRAIL_S * tracks.fps);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        for (const [sid, t] of subjects) {
          const colour = subjectColor(sid, "video");
          for (let f = Math.max(1, frame - span); f <= frame; f++) {
            const x0 = t.cx[f - 1];
            const y0 = t.cy[f - 1];
            const x1 = t.cx[f];
            const y1 = t.cy[f];
            if (x0 == null || y0 == null || x1 == null || y1 == null) continue;
            const a = (f - (frame - span)) / span;
            ctx.globalAlpha = a * 0.35;
            ctx.strokeStyle = "#0b0b0a";
            ctx.lineWidth = 4 * dpr;
            ctx.beginPath();
            ctx.moveTo(x0 * sx, y0 * sy);
            ctx.lineTo(x1 * sx, y1 * sy);
            ctx.stroke();
            ctx.globalAlpha = a * 0.95;
            ctx.strokeStyle = colour;
            ctx.lineWidth = 2 * dpr;
            ctx.stroke();
          }
        }
        ctx.globalAlpha = 1;
      }

      if (p.layers.masks) p.overlay.draw(ctx, frame, { width: p.width, height: p.height }, { width: cv.width, height: cv.height }, dpr);

      if (p.layers.tags && tracks) {
        const locked = lockedFrom(p.events);
        const doubts = p.events.filter((e) => isOpenBlocker(e) && IDENTITY_KINDS.has(e.kind) && frame >= e.span[0] && frame <= e.span[1]);
        const uncertainNow = new Set(doubts.flatMap((e) => e.subjects));
        const masks = p.overlay.masksAt(frame);
        ctx.font = `600 ${11.5 * dpr}px ${font.current}`;
        ctx.textBaseline = "middle";
        const chips: Chip[] = [];
        for (const [sid, t] of subjects) {
          const cx = t.cx[frame];
          const cy = t.cy[frame];
          if (cx == null || cy == null) continue;
          const m = masks.find((q) => q.subject === sid);
          const top = m ? m.rec.top : cy - 14;
          const isLocked = (locked.get(sid) ?? Infinity) <= frame;
          const text = subjectTag(sid, p.label);
          const w = ctx.measureText(text).width + (isLocked ? 22 : 12) * dpr;
          const h = 19 * dpr;
          chips.push({
            text,
            x: cx * sx - w / 2,
            y: top * sy - h - 6 * dpr,
            w,
            h,
            fill: uncertainNow.has(sid) ? null : subjectColor(sid, "video"),
            locked: isLocked,
            uncertain: uncertainNow.has(sid),
            focus: p.focusSubject === sid,
          });
        }
        // one label above the animals whose identities are in question right now
        const labels: Chip[] = [];
        ctx.font = `500 ${11.5 * dpr}px ${font.current}`;
        const involved = chips.filter((c) => c.uncertain);
        if (involved.length >= 2) {
          const text = "ID uncertain";
          const w = ctx.measureText(text).width + 14 * dpr;
          const h = 20 * dpr;
          const x = involved.reduce((a, c) => a + c.x + c.w / 2, 0) / involved.length - w / 2;
          const y = Math.min(...involved.map((c) => c.y)) - h - 6 * dpr;
          labels.push({ text, x, y, w, h, fill: "#0e0e0c", locked: false, uncertain: false, focus: false });
        }
        ctx.font = `600 ${11.5 * dpr}px ${font.current}`;
        const all = [...chips, ...labels];
        for (let it = 0; it < 6; it++)
          for (let a = 0; a < all.length; a++)
            for (let b = a + 1; b < all.length; b++) {
              const A = all[a]!;
              const B = all[b]!;
              const ox = Math.min(A.x + A.w, B.x + B.w) - Math.max(A.x, B.x) + 3 * dpr;
              const oy = Math.min(A.y + A.h, B.y + B.h) - Math.max(A.y, B.y) + 3 * dpr;
              if (ox > 0 && oy > 0) {
                if (ox < oy) {
                  const dir = A.x <= B.x ? 1 : -1;
                  A.x -= (dir * ox) / 2;
                  B.x += (dir * ox) / 2;
                } else {
                  const dir = A.y <= B.y ? 1 : -1;
                  A.y -= (dir * oy) / 2;
                  B.y += (dir * oy) / 2;
                }
              }
            }
        if (!hatch.current) {
          const hp = document.createElement("canvas");
          hp.width = hp.height = 6;
          const hc = hp.getContext("2d")!;
          hc.strokeStyle = "rgba(14,14,12,.3)";
          hc.lineWidth = 1;
          hc.beginPath();
          hc.moveTo(0, 6);
          hc.lineTo(6, 0);
          hc.moveTo(-1, 1);
          hc.lineTo(1, -1);
          hc.moveTo(5, 7);
          hc.lineTo(7, 5);
          hc.stroke();
          hatch.current = ctx.createPattern(hp, "repeat");
        }
        for (const c of chips) {
          c.x = Math.min(Math.max(c.x, 4 * dpr), cv.width - c.w - 4 * dpr);
          c.y = Math.min(Math.max(c.y, 4 * dpr), cv.height - c.h - 4 * dpr);
          ctx.beginPath();
          ctx.roundRect(c.x, c.y, c.w, c.h, 5 * dpr);
          if (c.fill) {
            ctx.fillStyle = c.fill;
            ctx.fill();
          } else {
            ctx.fillStyle = "#e9e7e2";
            ctx.fill();
            if (hatch.current) {
              ctx.fillStyle = hatch.current;
              ctx.fill();
            }
            ctx.strokeStyle = "rgba(14,14,12,.55)";
            ctx.lineWidth = 1 * dpr;
            ctx.stroke();
          }
          if (c.focus) {
            ctx.strokeStyle = "#2448e8";
            ctx.lineWidth = 2 * dpr;
            ctx.beginPath();
            ctx.roundRect(c.x - 3 * dpr, c.y - 3 * dpr, c.w + 6 * dpr, c.h + 6 * dpr, 7 * dpr);
            ctx.stroke();
          }
          ctx.fillStyle = "#0e0e0c";
          ctx.fillText(c.text, c.x + 6 * dpr, c.y + c.h / 2 + 0.5 * dpr);
          if (c.locked) {
            const gx = c.x + c.w - 13 * dpr;
            const gy = c.y + c.h / 2 - 1 * dpr;
            ctx.fillRect(gx, gy, 7 * dpr, 5.5 * dpr);
            ctx.beginPath();
            ctx.arc(gx + 3.5 * dpr, gy, 2.3 * dpr, Math.PI, 0);
            ctx.strokeStyle = "#0e0e0c";
            ctx.lineWidth = 1.2 * dpr;
            ctx.stroke();
          }
        }
        ctx.font = `500 ${11.5 * dpr}px ${font.current}`;
        for (const l of labels) {
          l.x = Math.min(Math.max(l.x, 4 * dpr), cv.width - l.w - 4 * dpr);
          l.y = Math.min(Math.max(l.y, 4 * dpr), cv.height - l.h - 4 * dpr);
          ctx.beginPath();
          ctx.roundRect(l.x, l.y, l.w, l.h, 5 * dpr);
          ctx.fillStyle = "#0e0e0c";
          ctx.fill();
          ctx.fillStyle = "#ecebe6";
          ctx.fillText(l.text, l.x + 7 * dpr, l.y + l.h / 2 + 0.5 * dpr);
        }
      }

      const ring = (m: Mark, active: boolean, caption: string | null) => {
        const x = m.x * sx;
        const y = m.y * sy;
        ctx.beginPath();
        ctx.arc(x, y, 13 * dpr, 0, Math.PI * 2);
        ctx.lineWidth = 4 * dpr;
        ctx.strokeStyle = "rgba(255,255,255,.75)";
        ctx.stroke();
        ctx.lineWidth = 2 * dpr;
        ctx.strokeStyle = "#2448e8";
        ctx.stroke();
        if (active) {
          ctx.fillStyle = "rgba(36,72,232,.25)";
          ctx.fill();
        }
        ctx.font = `600 ${11.5 * dpr}px ${font.current}`;
        const text = caption ?? subjectTag(m.subject, p.label);
        const w = ctx.measureText(text).width + 12 * dpr;
        const h = 19 * dpr;
        ctx.beginPath();
        ctx.roundRect(x + 14 * dpr, y - h - 6 * dpr, w, h, 5 * dpr);
        ctx.fillStyle = "#2448e8";
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.textBaseline = "middle";
        ctx.fillText(text, x + 20 * dpr, y - h / 2 - 5.5 * dpr);
      };

      if (p.anchors && Math.abs(frame - p.anchors.frame) <= 1)
        for (const m of p.anchors.marks) ring(m, false, `${subjectTag(m.subject, p.label)} marked by ${p.anchors.by}`);
      if (p.marking && frame === p.marking.frame) for (const m of p.marking.marks) ring(m, m.subject === p.marking.active, null);
    };
  });

  // follow the decoder: draw on every presented frame
  useEffect(() => {
    const v = video.current;
    if (!v) return;
    const player = latest.current.player;
    player.attach(v);
    let handle = 0;
    let raf = 0;
    let alive = true;
    const hasRvfc = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
    const onFrame = (_now: number, meta: { mediaTime: number }) => {
      if (!alive) return;
      player.setFrame(player.frameAt(meta.mediaTime));
      draw.current();
      handle = v.requestVideoFrameCallback(onFrame);
    };
    const poll = () => {
      if (!alive) return;
      if (!v.paused) {
        player.setFrame(player.frameAt(v.currentTime - 0.5 / player.fps));
        draw.current();
      }
      raf = requestAnimationFrame(poll);
    };
    if (hasRvfc) handle = v.requestVideoFrameCallback(onFrame);
    else raf = requestAnimationFrame(poll);
    const onPlay = () => player.setPlaying(true);
    const onPause = () => {
      player.setPlaying(false);
      draw.current();
    };
    const onSeeked = () => draw.current();
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("ended", onPause);
    v.addEventListener("seeked", onSeeked);
    v.addEventListener("loadeddata", onSeeked);
    return () => {
      alive = false;
      if (hasRvfc) v.cancelVideoFrameCallback(handle);
      cancelAnimationFrame(raf);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("ended", onPause);
      v.removeEventListener("seeked", onSeeked);
      v.removeEventListener("loadeddata", onSeeked);
      player.attach(null);
    };
  }, []);

  // redraw when the frame changes while paused (scrubbing, stepping) and when inputs change
  useEffect(() => {
    const player = props.player;
    return player.subscribe(() => {
      if (!player.isPlaying()) requestAnimationFrame(() => draw.current());
    });
  }, [props.player]);

  useEffect(() => {
    draw.current();
  }, [props.version, props.layers, props.focusSubject, props.anchors, props.marking, props.events, props.tracks]);

  // keep the canvas backing store at device resolution
  useEffect(() => {
    const el = stage.current;
    const cv = canvas.current;
    if (!el || !cv) return;
    const ro = new ResizeObserver(() => {
      const r = cv.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      cv.width = Math.max(1, Math.round(r.width * dpr));
      cv.height = Math.max(1, Math.round(r.height * dpr));
      draw.current();
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const m = props.marking;
    if (!m) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * props.width;
    const y = ((e.clientY - r.top) / r.height) * props.height;
    m.onMark(Math.round(x), Math.round(y));
  }

  return (
    <div
      ref={stage}
      className={styles.stage}
      style={{ aspectRatio: `${props.width} / ${props.height}` }}
      data-marking={props.marking ? "" : undefined}
    >
      <video
        ref={video}
        className={styles.video}
        src={props.src}
        muted
        playsInline
        preload="auto"
        aria-label="Experiment video"
        onClick={() => props.player.toggle()}
      />
      <canvas
        ref={canvas}
        className={styles.overlay}
        aria-hidden="true"
        onPointerDown={onPointerDown}
        style={{ pointerEvents: props.marking ? "auto" : "none" }}
      />
      {props.children}
    </div>
  );
}
