// Mask overlay for the workspace video, drawn with Meta's SAM client libraries.
//
// The segmenter's ids are its own; a run's subjects are the engine's. tracks.json
// says, for every subject and frame, which SAM object id stands behind it
// (source[frame], -1 when absent or interpolated). The overlay re-keys every
// mask record by subject before handing the view to @meta-sam/graphics, so a
// colour follows the animal, not the segmenter's id. Frames from a repair frame
// on come from that repair's re-track, whose ids are the subject numbers.

import { SegmentationRenderer } from "@meta-sam/graphics";
import { decodeMaskToRaster, decodeMaskToSVGPath, type VideoSegmentationResult } from "@meta-sam/parser";

import { NEUTRAL, subjectColor } from "@/lib/palette";
import type { Span, TracksFile } from "@/lib/types";

import type { MaskRec } from "./messages";

export const MAIN = "main";
export const LIVE = "live";
export const repairKey = (frame: number) => `repair-${frame}`;

export interface SubjectMapping {
  tracks: TracksFile | null;
  /** Repairs in frame order; a null key means the re-track text is not available here. */
  repairs: { frame: number; key: string | null }[];
  /** Per subject, the spans where an open identity doubt makes its mask uncertain. */
  uncertain: Map<number, Span[]>;
}

export type MaskRole = "subject" | "uncertain" | "unassigned" | "raw";

export interface FrameMask {
  rec: MaskRec;
  identity: string;
  id: string;
  role: MaskRole;
  subject: number | null;
}

interface Target {
  width: number;
  height: number;
}

/** Frames retained by the renderers on each side of the playhead. */
const WINDOW = 360;

const inSpans = (spans: Span[] | undefined, f: number) => !!spans?.some(([a, b]) => f >= a && f <= b);

function colourFor(id: string): string {
  const n = Number(id.slice(1));
  switch (id[0]) {
    case "s":
      return subjectColor(n, "video");
    case "t":
      return subjectColor(n + 1, "video");
    case "u":
      return NEUTRAL.video;
    default:
      return "#9a978f";
  }
}

class Lru<V> {
  private map = new Map<string, V>();
  constructor(private limit: number) {}
  get(key: string): V | undefined {
    const v = this.map.get(key);
    if (v !== undefined) {
      this.map.delete(key);
      this.map.set(key, v);
    }
    return v;
  }
  set(key: string, v: V) {
    this.map.set(key, v);
    if (this.map.size > this.limit) {
      const first = this.map.keys().next().value;
      if (first !== undefined) this.map.delete(first);
    }
  }
  clear() {
    this.map.clear();
  }
}

export class MaskOverlay {
  private sources = new Map<string, Map<number, Map<number, MaskRec>>>();
  private frames = new Map<number, FrameMask[]>();
  private mapping: SubjectMapping = { tracks: null, repairs: [], uncertain: new Map() };
  private keyline: SegmentationRenderer | null = null;
  private hue: SegmentationRenderer | null = null;
  private records = new Map<number, unknown[]>();
  private sortedFrames: number[] = [];
  private view: VideoSegmentationResult | null = null;
  private window: [number, number] | null = null;
  private center = 0;
  private revision = 0;
  private widthKey = 0;
  private pending: Promise<void> = Promise.resolve();
  private paths = new Lru<Path2D>(256);
  private rasters = new Lru<Uint8Array>(256);
  private disposed = false;
  private listeners = new Set<() => void>();

  /** Called after the retained view changes, so the stage can redraw. */
  listen(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  get size(): number {
    let n = 0;
    for (const frames of this.sources.values()) for (const m of frames.values()) n += m.size;
    return n;
  }

  sourceSize(key: string): number {
    let n = 0;
    for (const m of this.sources.get(key)?.values() ?? []) n += m.size;
    return n;
  }

  hasSource(key: string): boolean {
    return this.sources.has(key);
  }

  addMasks(key: string, masks: MaskRec[], replace = false) {
    let frames = this.sources.get(key);
    if (!frames || replace) {
      frames = new Map();
      this.sources.set(key, frames);
    }
    for (const m of masks) {
      let byId = frames.get(m.frame);
      if (!byId) {
        byId = new Map();
        frames.set(m.frame, byId);
      }
      byId.set(m.objectId, m);
    }
  }

  dropSource(key: string) {
    this.sources.delete(key);
  }

  setMapping(mapping: SubjectMapping) {
    this.mapping = mapping;
  }

  /** Which recorded output a frame's masks come from, or null when it isn't available. */
  private sourceFor(frame: number): string | null {
    if (this.sources.has(LIVE)) return LIVE;
    let key: string | null = MAIN;
    for (const r of this.mapping.repairs) if (r.frame <= frame) key = r.key;
    return key;
  }

  /** Re-key every mask by subject and hand the view to the renderers. */
  rebuild(): Promise<void> {
    const frames = new Map<number, FrameMask[]>();
    const recordsByFrame = new Map<number, unknown[]>();
    const all = new Set<number>();
    for (const src of this.sources.values()) for (const f of src.keys()) all.add(f);
    const { tracks, uncertain } = this.mapping;
    const subjects = tracks ? Object.entries(tracks.subjects).map(([sid, t]) => [Number(sid), t.source] as const) : [];
    for (const f of [...all].sort((a, b) => a - b)) {
      const key = this.sourceFor(f);
      if (key === null) continue;
      const recs = this.sources.get(key)?.get(f);
      if (!recs) continue;
      const claimed = new Map<number, number>();
      if (key !== LIVE) for (const [sid, source] of subjects) {
        const s = source[f];
        if (s !== undefined && s >= 0) claimed.set(s, sid);
      }
      const list: FrameMask[] = [];
      const records: unknown[] = [];
      for (const rec of recs.values()) {
        let id: string;
        let role: MaskRole;
        let subject: number | null = null;
        if (key === LIVE || !tracks) {
          id = `t${rec.objectId}`;
          role = "raw";
        } else {
          const sid = claimed.get(rec.objectId);
          if (sid === undefined) {
            id = `x${rec.objectId}`;
            role = "unassigned";
          } else {
            subject = sid;
            role = inSpans(uncertain.get(sid), f) ? "uncertain" : "subject";
            id = `${role === "uncertain" ? "u" : "s"}${sid}`;
          }
        }
        const identity = `${key}:${f}:${rec.objectId}`;
        list.push({ rec, identity, id, role, subject });
        records.push({
          kind: "mask",
          order: 0,
          objectId: id,
          frame: { frameIndex: f },
          identity,
          revision: 1,
          mask: { encoding: rec.encoding, payload: rec.payload, width: rec.width, height: rec.height },
          bounds: { left: rec.left, top: rec.top, right: rec.right, bottom: rec.bottom },
        });
      }
      frames.set(f, list);
      recordsByFrame.set(f, records);
    }
    this.frames = frames;
    this.records = recordsByFrame;
    this.sortedFrames = [...recordsByFrame.keys()];
    return this.focus(this.center, true);
  }

  /**
   * The renderers retain a window of frames around the playhead, not the whole run:
   * a ten-minute run of five animals is 90,000 masks, and only the frame on screen is
   * ever traced. Moving outside the window hands them the next one.
   */
  private focus(frame: number, force = false): Promise<void> {
    this.center = frame;
    const w = this.window;
    if (!force && w && frame >= w[0] + WINDOW / 4 && frame <= w[1] - WINDOW / 4) return Promise.resolve();
    const lo = frame - WINDOW;
    const hi = frame + WINDOW;
    const records: unknown[] = [];
    let order = 0;
    for (const f of this.sortedFrames) {
      if (f < lo || f > hi) continue;
      for (const r of this.records.get(f) ?? []) records.push({ ...(r as object), order: order++ });
    }
    this.window = [lo, hi];
    this.view = {
      media: "video",
      revision: ++this.revision,
      records,
      diagnostics: [],
      rawOutput: "",
      outcome: { status: "completed" },
    } as unknown as VideoSegmentationResult;
    return this.push();
  }

  private push(): Promise<void> {
    const view = this.view;
    const keyline = this.keyline;
    const hue = this.hue;
    if (!view || !keyline || !hue) return Promise.resolve();
    this.pending = Promise.all([keyline.update(view, { reset: true }), hue.update(view, { reset: true })]).then(() => {
      if (!this.disposed) for (const fn of this.listeners) fn();
    });
    return this.pending;
  }

  /** Outline widths are fixed per renderer in source pixels; keep them near 2 screen pixels. */
  private ensureRenderers(sourcePerPx: number) {
    const key = Math.round(sourcePerPx * 8) / 8;
    if (this.keyline && this.hue && Math.abs(key - this.widthKey) / this.widthKey < 0.2) return;
    this.widthKey = key;
    this.keyline?.dispose();
    this.hue?.dispose();
    this.keyline = new SegmentationRenderer({
      maskFillOpacity: 0,
      maskOutline: { width: 4.4 * key, opacity: 0.62 },
      objectColor: () => "#0b0b0a",
      maxCachedPaths: 512,
      maxMasks: 16_384,
      maxRecords: 32_768,
      maxRetainedComplexity: 4_000_000,
    });
    this.hue = new SegmentationRenderer({
      maskFillOpacity: 0.36,
      maskOutline: { width: 2 * key, opacity: 1 },
      objectColor: colourFor,
      maxCachedPaths: 512,
      maxMasks: 16_384,
      maxRecords: 32_768,
      maxRetainedComplexity: 4_000_000,
    });
    void this.push();
  }

  masksAt(frame: number): FrameMask[] {
    return this.frames.get(frame) ?? [];
  }

  /** Draw one frame's masks. `source` is the video's pixel size, `target` the canvas backing size. */
  draw(ctx: CanvasRenderingContext2D, frame: number, source: Target, target: Target, dpr: number) {
    const scale = target.width / source.width;
    this.ensureRenderers(dpr / scale);
    void this.focus(frame);
    const list = this.frames.get(frame);
    if (!list || !this.keyline || !this.hue) return;
    const hidden = list.filter((m) => m.role === "uncertain").map((m) => m.id);
    const opts = {
      media: "video" as const,
      frameIndex: frame,
      source: { x: 0, y: 0, width: source.width, height: source.height },
      target: { x: 0, y: 0, width: target.width, height: target.height },
      hiddenIds: hidden,
    };
    try {
      this.keyline.render(ctx, opts);
      this.hue.render(ctx, opts);
    } catch {
      // a frame whose view is mid-update draws on the next tick
    }
    // uncertain masks: grey and dashed, so a doubted identity never looks settled
    for (const m of list) {
      if (m.role !== "uncertain") continue;
      const path = this.path(m);
      if (!path) continue;
      ctx.save();
      ctx.scale(scale, scale);
      ctx.translate(m.rec.left, m.rec.top);
      ctx.scale((m.rec.right - m.rec.left) / m.rec.width, (m.rec.bottom - m.rec.top) / m.rec.height);
      ctx.globalAlpha = 0.42;
      ctx.fillStyle = NEUTRAL.video;
      ctx.fill(path, "evenodd");
      ctx.globalAlpha = 1;
      ctx.lineJoin = "round";
      ctx.setLineDash([4 * (dpr / scale), 3 * (dpr / scale)]);
      ctx.lineWidth = 3.6 * (dpr / scale);
      ctx.strokeStyle = "rgba(11,11,10,.55)";
      ctx.stroke(path);
      ctx.lineWidth = 1.8 * (dpr / scale);
      ctx.strokeStyle = "#ecebe6";
      ctx.stroke(path);
      ctx.restore();
    }
  }

  private path(m: FrameMask): Path2D | null {
    const cached = this.paths.get(m.identity);
    if (cached) return cached;
    try {
      const d = decodeMaskToSVGPath({ encoding: m.rec.encoding, payload: m.rec.payload, width: m.rec.width, height: m.rec.height });
      const p = new Path2D(d);
      this.paths.set(m.identity, p);
      return p;
    } catch {
      return null;
    }
  }

  private raster(m: FrameMask): Uint8Array | null {
    const cached = this.rasters.get(m.identity);
    if (cached) return cached;
    try {
      const r = decodeMaskToRaster({ encoding: m.rec.encoding, payload: m.rec.payload, width: m.rec.width, height: m.rec.height });
      this.rasters.set(m.identity, r);
      return r;
    } catch {
      return null;
    }
  }

  /** The mask under a point in video pixels on a frame, decoded on demand. */
  hitTest(frame: number, x: number, y: number): FrameMask | null {
    for (const m of this.frames.get(frame) ?? []) {
      const { left, top, right, bottom, width, height } = m.rec;
      if (x < left || x >= right || y < top || y >= bottom) continue;
      const raster = this.raster(m);
      if (!raster) continue;
      const col = Math.floor(((x - left) / (right - left)) * width);
      const row = Math.floor(((y - top) / (bottom - top)) * height);
      if (raster[row * width + col]) return m;
    }
    return null;
  }

  dispose() {
    this.disposed = true;
    this.keyline?.dispose();
    this.hue?.dispose();
    this.paths.clear();
    this.rasters.clear();
    this.sources.clear();
    this.frames.clear();
  }
}
