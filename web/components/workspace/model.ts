import type { EventView, MetricSpec, ResultView, RunData, Span } from "@/lib/types";

export const IDENTITY_KINDS = new Set(["contact", "id_churn"]);

export const isOpenBlocker = (e: EventView) => e.severity === "blocking" && !e.resolved_by;

export function resultKey(r: ResultView): string {
  return `${r.metric_id}|${r.subject ?? "all"}|${r.bin_frames ? r.bin_frames.join("-") : "all"}`;
}

export function subjectsOf(run: RunData, trackIds: number[]): number[] {
  const set = new Set<number>(trackIds);
  for (const r of run.results) if (r.subject != null) set.add(r.subject);
  for (const e of run.events) for (const s of e.subjects) set.add(s);
  if (!set.size && run.protocol) for (let i = 1; i <= run.protocol.subject.expected_count; i++) set.add(i);
  return [...set].sort((a, b) => a - b);
}

/** Spans where a subject's identity is in doubt right now: inside an open identity event. */
export function uncertainSpans(events: EventView[]): Map<number, Span[]> {
  const out = new Map<number, Span[]>();
  for (const e of events) {
    if (!isOpenBlocker(e) || !IDENTITY_KINDS.has(e.kind)) continue;
    for (const s of e.subjects) {
      const list = out.get(s) ?? [];
      list.push(e.span);
      out.set(s, list);
    }
  }
  return out;
}

/** The first frame from which each subject's identity stays unconfirmed (an open identity event locks to the end). */
export function lockedFrom(events: EventView[]): Map<number, number> {
  const out = new Map<number, number>();
  for (const e of events) {
    if (!isOpenBlocker(e) || !IDENTITY_KINDS.has(e.kind)) continue;
    for (const s of e.subjects) out.set(s, Math.min(out.get(s) ?? Infinity, e.span[0]));
  }
  return out;
}

export interface MetricGroup {
  id: string;
  spec: MetricSpec | undefined;
  unit: string;
  rows: { subject: number | null; whole: ResultView | null; bins: ResultView[] }[];
}

/** Results grouped by metric (in protocol order), then by subject, with bins under each subject. */
export function groupResults(run: RunData): MetricGroup[] {
  const order = run.protocol?.metrics.map((m) => m.id) ?? [];
  const ids = [...new Set([...order, ...run.results.map((r) => r.metric_id)])].filter((id) =>
    run.results.some((r) => r.metric_id === id),
  );
  return ids.map((id) => {
    const results = run.results.filter((r) => r.metric_id === id);
    const subjects = [...new Set(results.map((r) => r.subject))].sort((a, b) => (a ?? 0) - (b ?? 0));
    return {
      id,
      spec: run.protocol?.metrics.find((m) => m.id === id),
      unit: results[0]?.unit ?? "",
      rows: subjects.map((subject) => {
        const mine = results.filter((r) => r.subject === subject);
        return {
          subject,
          whole: mine.find((r) => !r.bin_frames) ?? null,
          bins: mine.filter((r) => r.bin_frames).sort((a, b) => a.bin_frames![0] - b.bin_frames![0]),
        };
      }),
    };
  });
}

export function binEdges(run: RunData): number[] {
  const edges = new Set<number>();
  for (const r of run.results) if (r.bin_frames) edges.add(r.bin_frames[0]);
  edges.delete(0);
  return [...edges].sort((a, b) => a - b);
}

export function toCsv(run: RunData): string {
  const esc = (v: unknown) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const head = ["run_id", "protocol_sha256", "metric_id", "subject", "bin_start_s", "bin_end_s", "status", "value", "unit", "uncertainty", "fix_frame", "blocked_by", "reasons"];
  const rows = run.results.map((r) => [
    run.id,
    run.protocolSha256 ?? "",
    r.metric_id,
    r.subject ?? "all",
    r.bin_s?.[0] ?? "",
    r.bin_s?.[1] ?? "",
    r.status,
    r.value ?? "",
    r.unit,
    Object.keys(r.uncertainty).length ? JSON.stringify(r.uncertainty) : "",
    r.fix_frame ?? "",
    r.blocked_by.join(" "),
    r.reasons.join(" | "),
  ]);
  return [head, ...rows].map((row) => row.map(esc).join(",")).join("\n") + "\n";
}
