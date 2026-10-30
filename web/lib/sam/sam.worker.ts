// Parses SAM 3.1's raw output text off the main thread with Meta's parser.
// A recorded run's sam.txt is the concatenated output_text lane: one line per frame.
// Live tracking streams the same lines as they arrive.

import { formats, recordsOfKind, type SegmentationMaskRecord } from "@meta-sam/parser";

import type { FromWorker, MaskRec, ToWorker } from "./messages";

const scope = self as unknown as {
  postMessage(message: FromWorker): void;
  onmessage: ((event: MessageEvent<ToWorker>) => void) | null;
};

type Parser = ReturnType<ReturnType<typeof formats.segmentation.video>["createParser"]>;
const streams = new Map<string, { parser: Parser; sent: number }>();

function toRec(r: SegmentationMaskRecord): MaskRec | null {
  const frame = r.frame?.frameIndex;
  const id = Number(r.objectId);
  if (frame === undefined || !Number.isFinite(id)) return null;
  return {
    frame,
    objectId: id,
    left: r.bounds.left,
    top: r.bounds.top,
    right: r.bounds.right,
    bottom: r.bounds.bottom,
    encoding: r.mask.encoding,
    payload: r.mask.payload,
    width: r.mask.width,
    height: r.mask.height,
    ...(r.confidence === undefined ? {} : { confidence: r.confidence }),
  };
}

/** The latest revision of each (frame, object) mask, in output order. */
function latest(records: readonly SegmentationMaskRecord[]): MaskRec[] {
  const byKey = new Map<string, SegmentationMaskRecord>();
  for (const r of records) {
    const key = r.identity;
    const prev = byKey.get(key);
    if (!prev || r.revision >= prev.revision) byKey.set(key, r);
  }
  const out: MaskRec[] = [];
  for (const r of byKey.values()) {
    const rec = toRec(r);
    if (rec) out.push(rec);
  }
  return out;
}

function post(msg: FromWorker) {
  scope.postMessage(msg);
}

async function load(key: string, url: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const text = await res.text();
  const lines = text.split("\n");
  const parser = formats.segmentation.video().createParser();
  const CHUNK = 120;
  for (let i = 0; i < lines.length; i += CHUNK) {
    const chunk = lines.slice(i, i + CHUNK).join("\n") + (i + CHUNK < lines.length ? "\n" : "");
    parser.push(chunk, { emit: false });
    post({ type: "progress", key, lines: Math.min(lines.length, i + CHUNK), total: lines.length });
  }
  const { result } = parser.finish({ status: "completed" });
  post({ type: "masks", key, masks: latest(recordsOfKind(result.records, "mask")), done: true, diagnostics: result.diagnostics.length });
}

function delta(key: string, text: string) {
  const s = streams.get(key);
  if (!s) return;
  const events = s.parser.push(text, { emit: true });
  const snap = events.at(-1);
  if (!snap) return;
  const fresh = snap.records.slice(s.sent);
  s.sent = snap.records.length;
  const masks = recordsOfKind(fresh, "mask")
    .map(toRec)
    .filter((m): m is MaskRec => m !== null);
  if (masks.length) post({ type: "masks", key, masks, done: false, diagnostics: snap.diagnostics.length });
}

scope.onmessage = (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  try {
    if (msg.type === "load") {
      load(msg.key, msg.url).catch((error: unknown) => post({ type: "error", key: msg.key, message: String(error instanceof Error ? error.message : error) }));
    } else if (msg.type === "stream-start") {
      streams.set(msg.key, { parser: formats.segmentation.video().createParser(), sent: 0 });
    } else if (msg.type === "stream-delta") {
      delta(msg.key, msg.text);
    } else if (msg.type === "stream-end") {
      const s = streams.get(msg.key);
      if (s) {
        const { result } = s.parser.finish({ status: "completed" });
        const fresh = recordsOfKind(result.records.slice(s.sent), "mask")
          .map(toRec)
          .filter((m): m is MaskRec => m !== null);
        post({ type: "masks", key: msg.key, masks: fresh, done: true, diagnostics: result.diagnostics.length });
        streams.delete(msg.key);
      }
    }
  } catch (error) {
    post({ type: "error", key: msg.key, message: String(error instanceof Error ? error.message : error) });
  }
};
