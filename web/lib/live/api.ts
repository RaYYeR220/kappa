// The live workspace talks to the Kappa HTTP API (engine/src/kappa/api.py) when
// NEXT_PUBLIC_KAPPA_API is set. Reads are public; writes carry the operator token,
// which the person enters and which lives only in this tab's sessionStorage.
// Job progress streams as server-sent events, read with fetch so the request can
// carry the Authorization header (EventSource cannot).

import type {
  EventView,
  Protocol,
  RepairView,
  ResultView,
  Review,
  RunData,
  Verdict,
  VideoInfo,
} from "@/lib/types";

export const API_BASE = (process.env.NEXT_PUBLIC_KAPPA_API ?? "").replace(/\/+$/, "") || null;

const TOKEN_KEY = "kappa.operatorToken";

export function readToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function writeToken(token: string) {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // storage blocked: the token lasts as long as the page
  }
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function url(path: string): string {
  if (!API_BASE) throw new ApiError(0, "No Kappa API is configured for this site.");
  return `${API_BASE}${path}`;
}

async function detail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail ?? body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url(path), { signal, cache: "no-store" });
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return (await res.json()) as T;
}

async function post<T>(path: string, token: string, body?: unknown): Promise<T> {
  const res = await fetch(url(path), {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return (await res.json()) as T;
}

export async function health(): Promise<{ ok: boolean; public: boolean }> {
  return get("/health");
}

interface StateBody {
  run_id: string;
  created_at: string;
  source_name: string;
  video_sha256: string;
  video: VideoInfo;
  question: string | null;
  assumptions: string[];
  protocol: Protocol | null;
  confirmed_by: string | null;
  repairs: { frame: number; by: string; anchors: RepairView["anchors"] }[];
  behaviours: Record<string, unknown>;
  review: Review | null;
  manifest_sha256: string | null;
  confirmed: boolean;
  tracked: boolean;
}

export interface LiveRunRow {
  id: string;
  createdAt: string;
  sourceName: string;
  title: string;
  confirmed: boolean;
  tracked: boolean;
  frames: number;
  fps: number;
}

export async function listRuns(signal?: AbortSignal): Promise<LiveRunRow[]> {
  const rows = await get<StateBody[]>("/runs", signal);
  return rows.map((s) => ({
    id: s.run_id,
    createdAt: s.created_at,
    sourceName: s.source_name,
    title: s.protocol?.title ?? s.source_name,
    confirmed: s.confirmed,
    tracked: s.tracked,
    frames: s.video.n_frames,
    fps: s.video.fps,
  }));
}

export async function loadRun(id: string, signal?: AbortSignal): Promise<RunData> {
  const enc = encodeURIComponent(id);
  const state = await get<StateBody>(`/runs/${enc}`, signal);
  type Results = { protocol_sha256: string; analysed: [number, number]; results: ResultView[]; events: EventView[] };
  let results: Results | null = null;
  let manifest: Record<string, unknown> | null = null;
  if (state.tracked && state.confirmed) {
    results = await get<Results>(`/runs/${enc}/results`, signal).catch(() => null);
    manifest = await get<Record<string, unknown>>(`/runs/${enc}/manifest`, signal).catch(() => null);
  }
  const verdicts: Record<string, Verdict[]> = {};
  for (const b of Object.keys(state.behaviours ?? {})) {
    const v = await get<Verdict[]>(`/runs/${enc}/verdicts/${encodeURIComponent(b)}`, signal).catch(() => null);
    if (v) verdicts[b] = v;
  }
  const review = (manifest?.review as Review[] | undefined) ?? (state.review ? [state.review] : []);
  return {
    id: state.run_id,
    origin: "live",
    title: state.protocol?.title ?? state.source_name,
    question: state.question ?? state.protocol?.question ?? null,
    sourceName: state.source_name,
    video: state.video,
    videoSha256: state.video_sha256,
    protocol: state.protocol,
    protocolSha256: results?.protocol_sha256 ?? null,
    confirmedBy: state.confirmed_by,
    assumptions: state.assumptions ?? [],
    // the API serves the main tracking output only; repaired frames show positions without masks
    repairs: (state.repairs ?? []).map((r) => ({ frame: r.frame, by: r.by, anchors: r.anchors ?? [], sam: null })).sort((a, b) => a.frame - b.frame),
    corners: state.protocol?.arena.corners_px ?? null,
    analysed: results?.analysed ?? null,
    results: results?.results ?? [],
    events: results?.events ?? [],
    verdicts,
    manifest: manifest
      ? {
          createdAt: (manifest.created_at as string) ?? null,
          engineVersion: (manifest.engine_version as string) ?? null,
          models: (manifest.models as Record<string, string>) ?? {},
          review,
          methods: (manifest.methods as string) ?? null,
          withheld: (manifest.withheld as string) ?? null,
        }
      : review.length
        ? { createdAt: null, engineVersion: null, models: {}, review, methods: null, withheld: null }
        : null,
    tracked: state.tracked,
    urls: {
      video: url(`/runs/${enc}/video`),
      sam: state.tracked ? url(`/runs/${enc}/sam`) : null,
      tracks: state.tracked && state.confirmed ? url(`/runs/${enc}/tracks`) : null,
      manifest: manifest ? url(`/runs/${enc}/manifest`) : null,
    },
  };
}

export interface JobEvent {
  type: string;
  delta?: string;
  message?: string;
  [key: string]: unknown;
}

export async function startTrack(id: string, token: string) {
  return post<{ job: string; stream: string }>(`/runs/${encodeURIComponent(id)}/track`, token);
}

export async function startRepair(
  id: string,
  token: string,
  body: { frame: number; by: string; anchors: { subject: number; x: number; y: number }[] },
) {
  return post<{ job: string; stream: string }>(`/runs/${encodeURIComponent(id)}/repair`, token, body);
}

export async function startReview(id: string, token: string) {
  return post<{ job: string; stream: string }>(`/runs/${encodeURIComponent(id)}/review`, token);
}

/** Follow a job's event stream until it reports done or error. */
export async function* follow(stream: string, token: string, signal?: AbortSignal): AsyncGenerator<JobEvent> {
  const res = await fetch(url(stream), {
    headers: { Authorization: `Bearer ${token}`, Accept: "text/event-stream" },
    signal,
    cache: "no-store",
  });
  if (!res.ok || !res.body) throw new ApiError(res.status, await detail(res));
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += value;
      let cut: number;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const data = block
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).replace(/^ /, ""))
          .join("\n");
        if (!data) continue; // keep-alive comment
        const event = JSON.parse(data) as JobEvent;
        yield event;
        if (event.type === "done" || event.type === "error") return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
