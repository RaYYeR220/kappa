import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

import type {
  EventView,
  ManifestSummary,
  Protocol,
  RepairView,
  ResultView,
  Review,
  RunCard,
  RunData,
  RunOrigin,
  Status,
  TracksFile,
  Verdict,
  VideoInfo,
} from "./types";

/**
 * Static bundles: public/runs/<id>/ (published) and, when NEXT_PUBLIC_DEV_RUNS=1,
 * public/runs-dev/<id>/ (development fixtures, never published).
 *
 * Each index file is {"runs": [{"id": "<run id>"}]}; the order is the order shown.
 */

const PUBLIC = path.join(process.cwd(), "public");
const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;

export const devRunsEnabled = process.env.NEXT_PUBLIC_DEV_RUNS === "1";

const roots: { origin: Exclude<RunOrigin, "live">; dir: string; url: string }[] = [
  { origin: "published", dir: "runs", url: "/runs" },
  ...(devRunsEnabled ? [{ origin: "dev" as const, dir: "runs-dev", url: "/runs-dev" }] : []),
];

async function readJson<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(file, "utf8")) as T;
  } catch {
    return null;
  }
}

async function exists(file: string): Promise<boolean> {
  try {
    await readFile(file, { flag: "r" });
    return true;
  } catch {
    return false;
  }
}

/** Every bundled run id with where it lives, published runs first. */
export async function listRunIds(): Promise<{ id: string; origin: Exclude<RunOrigin, "live"> }[]> {
  const out: { id: string; origin: Exclude<RunOrigin, "live"> }[] = [];
  const seen = new Set<string>();
  for (const root of roots) {
    const index = await readJson<{ runs?: { id?: unknown }[] }>(path.join(PUBLIC, root.dir, "index.json"));
    for (const entry of index?.runs ?? []) {
      const id = typeof entry?.id === "string" ? entry.id : null;
      if (!id || !RUN_ID.test(id) || seen.has(id)) continue;
      seen.add(id);
      out.push({ id, origin: root.origin });
    }
  }
  return out;
}

interface RunFile {
  run_id: string;
  title?: string;
  question?: string | null;
  source_name: string;
  video: VideoInfo;
  video_sha256: string;
  protocol: Protocol | null;
  protocol_sha256?: string | null;
  confirmed_by?: string | null;
  assumptions?: string[];
  repairs?: { frame: number; by: string; anchors: RepairView["anchors"]; file?: string }[];
  corners_px?: [number, number][] | null;
}

interface ResultsFile {
  analysed: [number, number];
  results: ResultView[];
  events: EventView[];
}

interface ManifestFile {
  created_at?: string;
  engine_version?: string;
  models?: Record<string, string>;
  review?: Review[];
  methods?: string;
  withheld?: string;
  summary?: Record<Status, number>;
}

function summarise(results: ResultView[]): Record<Status, number> {
  const counts: Record<Status, number> = { verified: 0, locked: 0, refused: 0 };
  for (const r of results) counts[r.status] += 1;
  return counts;
}

function locate(id: string, origin: Exclude<RunOrigin, "live">) {
  const root = roots.find((r) => r.origin === origin);
  if (!root) return null;
  return { dir: path.join(PUBLIC, root.dir, id), url: `${root.url}/${id}` };
}

export async function loadRun(id: string): Promise<RunData | null> {
  if (!RUN_ID.test(id)) return null;
  const entry = (await listRunIds()).find((r) => r.id === id);
  if (!entry) return null;
  const where = locate(id, entry.origin);
  if (!where) return null;

  const [run, results, manifest, verdicts] = await Promise.all([
    readJson<RunFile>(path.join(where.dir, "run.json")),
    readJson<ResultsFile>(path.join(where.dir, "results.json")),
    readJson<ManifestFile>(path.join(where.dir, "manifest.json")),
    readJson<Record<string, Verdict[]>>(path.join(where.dir, "verdicts.json")),
  ]);
  if (!run || !results) return null;

  const repairs: RepairView[] = [];
  for (const r of run.repairs ?? []) {
    const file = r.file ?? `sam-repair-${r.frame}.txt`;
    repairs.push({
      frame: r.frame,
      by: r.by,
      anchors: r.anchors ?? [],
      sam: (await exists(path.join(where.dir, file))) ? `${where.url}/${file}` : null,
    });
  }

  const summary: ManifestSummary | null = manifest
    ? {
        createdAt: manifest.created_at ?? null,
        engineVersion: manifest.engine_version ?? null,
        models: manifest.models ?? {},
        review: manifest.review ?? [],
        methods: manifest.methods ?? null,
        withheld: manifest.withheld ?? null,
      }
    : null;

  return {
    id,
    origin: entry.origin,
    title: run.title || run.protocol?.title || id,
    question: run.question ?? run.protocol?.question ?? null,
    sourceName: run.source_name,
    video: run.video,
    videoSha256: run.video_sha256,
    protocol: run.protocol,
    protocolSha256: run.protocol_sha256 ?? null,
    confirmedBy: run.confirmed_by ?? null,
    assumptions: run.assumptions ?? [],
    repairs: repairs.sort((a, b) => a.frame - b.frame),
    corners: run.corners_px ?? run.protocol?.arena.corners_px ?? null,
    analysed: results.analysed,
    results: results.results,
    events: results.events,
    verdicts: verdicts ?? {},
    manifest: summary,
    tracked: true,
    urls: {
      video: `${where.url}/video.mp4`,
      sam: `${where.url}/sam.txt`,
      tracks: `${where.url}/tracks.json`,
      manifest: manifest ? `${where.url}/manifest.json` : null,
    },
  };
}

/** Thumbnail paths: each subject's centroid track, every few frames, gaps dropped. */
function samplePaths(tracks: TracksFile | null, step: number): RunCard["paths"] {
  if (!tracks) return [];
  return Object.entries(tracks.subjects).map(([sid, t]) => {
    const points: [number, number][] = [];
    for (let f = 0; f < t.cx.length; f += step) {
      const x = t.cx[f];
      const y = t.cy[f];
      if (x != null && y != null) points.push([Math.round(x), Math.round(y)]);
    }
    return { subject: Number(sid), points };
  });
}

export async function listRunCards(): Promise<RunCard[]> {
  const cards: RunCard[] = [];
  for (const { id, origin } of await listRunIds()) {
    const where = locate(id, origin);
    if (!where) continue;
    const [run, results, manifest, tracks] = await Promise.all([
      readJson<RunFile>(path.join(where.dir, "run.json")),
      readJson<ResultsFile>(path.join(where.dir, "results.json")),
      readJson<ManifestFile>(path.join(where.dir, "manifest.json")),
      readJson<TracksFile>(path.join(where.dir, "tracks.json")),
    ]);
    if (!run || !results) continue;
    const step = Math.max(1, Math.round(run.video.fps / 6));
    cards.push({
      id,
      origin,
      title: run.title || run.protocol?.title || id,
      question: run.question ?? run.protocol?.question ?? null,
      subjectLabel: run.protocol?.subject.label ?? "subject",
      subjectCount: run.protocol?.subject.expected_count ?? Object.keys(tracks?.subjects ?? {}).length,
      durationS: run.video.duration_s,
      fps: run.video.fps,
      frames: run.video.n_frames,
      summary: manifest?.summary ?? summarise(results.results),
      confirmedBy: run.confirmed_by ?? null,
      createdAt: manifest?.created_at ?? null,
      paths: samplePaths(tracks, step),
      width: run.video.width,
      height: run.video.height,
    });
  }
  return cards;
}

/** Where the landing page's main call to action goes: the first published run, else the list. */
export async function firstRunHref(): Promise<string> {
  const ids = await listRunIds();
  const first = ids.find((r) => r.origin === "published") ?? ids[0];
  return first ? `/runs/${first.id}` : "/runs";
}
