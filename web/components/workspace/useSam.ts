"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { FromWorker, ToWorker } from "@/lib/sam/messages";
import { LIVE, MAIN, MaskOverlay, repairKey } from "@/lib/sam/overlay";
import type { EventView, RepairView, TracksFile } from "@/lib/types";

import { uncertainSpans } from "./model";

export interface SamStatus {
  state: "idle" | "loading" | "ready" | "streaming" | "error";
  loaded: number;
  total: number;
  masks: number;
  message?: string;
}

/**
 * Owns the parsing worker and the mask overlay for one run. Recorded output is
 * fetched and parsed in the worker; live tracking output is streamed into it.
 */
export function useSam(samUrl: string | null, repairs: RepairView[], tracks: TracksFile | null, events: EventView[]) {
  const [overlay] = useState(() => new MaskOverlay());
  const [version, setVersion] = useState(0);
  const [status, setStatus] = useState<SamStatus>(() => ({
    state: samUrl ? "loading" : "idle",
    loaded: 0,
    total: 0,
    masks: 0,
  }));
  const worker = useRef<Worker | null>(null);
  const rebuildTimer = useRef<number>(0);
  const streaming = useRef(false);

  // streamed masks arrive many times a second: rebuild at most every 150 ms, and only while streaming
  const onLive = useRef((masks: FromWorker & { type: "masks" }) => {
    if (!streaming.current) return;
    overlay.addMasks(LIVE, masks.masks);
    if (!rebuildTimer.current)
      rebuildTimer.current = window.setTimeout(() => {
        rebuildTimer.current = 0;
        void overlay.rebuild();
      }, 150);
    setStatus((s) => ({ ...s, masks: overlay.sourceSize(LIVE) }));
  });
  const repairSig = repairs.map((r) => `${r.frame}:${r.sam ?? ""}`).join("|");

  // mapping from segmenter ids to subjects follows the tracks and the open identity doubts
  useEffect(() => {
    overlay.setMapping({
      tracks,
      repairs: repairs.map((r) => ({ frame: r.frame, key: r.sam ? repairKey(r.frame) : null })),
      uncertain: uncertainSpans(events),
    });
    void overlay.rebuild();
    // repairSig stands in for repairs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlay, tracks, events, repairSig]);

  useEffect(() => overlay.listen(() => setVersion((v) => v + 1)), [overlay]);

  useEffect(() => {
    if (!samUrl) return;
    worker.current?.terminate();
    const w = new Worker(new URL("../../lib/sam/sam.worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    const wanted = new Set<string>([MAIN]);
    const loads: ToWorker[] = [{ type: "load", key: MAIN, url: samUrl }];
    for (const r of repairs) {
      if (!r.sam) continue;
      wanted.add(repairKey(r.frame));
      loads.push({ type: "load", key: repairKey(r.frame), url: r.sam });
    }
    const done = new Set<string>();
    const progress = new Map<string, [number, number]>();

    w.onmessage = (event: MessageEvent<FromWorker>) => {
      const msg = event.data;
      if (msg.type === "progress") {
        progress.set(msg.key, [msg.lines, msg.total]);
        let loaded = 0;
        let total = 0;
        for (const [l, t] of progress.values()) {
          loaded += l;
          total += t;
        }
        setStatus((s) => (s.state === "streaming" ? s : { ...s, state: "loading", loaded, total }));
      } else if (msg.type === "masks") {
        if (msg.key === LIVE) {
          onLive.current(msg);
          return;
        }
        overlay.addMasks(msg.key, msg.masks);
        if (msg.done) done.add(msg.key);
        if ([...wanted].every((k) => done.has(k))) {
          void overlay.rebuild();
          setStatus({ state: "ready", loaded: 1, total: 1, masks: overlay.size });
        }
      } else if (msg.type === "error") {
        setStatus((s) => ({ ...s, state: "error", message: msg.message }));
      }
    };
    for (const m of loads) w.postMessage(m);
    return () => {
      w.terminate();
      worker.current = null;
    };
    // repairSig stands in for repairs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samUrl, repairSig, overlay]);

  const stream = {
    start: useCallback(() => {
      let w = worker.current;
      if (!w) {
        w = new Worker(new URL("../../lib/sam/sam.worker.ts", import.meta.url), { type: "module" });
        worker.current = w;
        w.onmessage = (event: MessageEvent<FromWorker>) => {
          const msg = event.data;
          if (msg.type === "masks" && msg.key === LIVE) {
            onLive.current(msg);
          } else if (msg.type === "error") {
            setStatus((s) => ({ ...s, state: "error", message: msg.message }));
          }
        };
      }
      streaming.current = true;
      overlay.addMasks(LIVE, [], true);
      w.postMessage({ type: "stream-start", key: LIVE } satisfies ToWorker);
      setStatus({ state: "streaming", loaded: 0, total: 0, masks: 0 });
    }, [overlay]),
    push: useCallback((text: string) => {
      worker.current?.postMessage({ type: "stream-delta", key: LIVE, text } satisfies ToWorker);
    }, []),
    end: useCallback(() => {
      worker.current?.postMessage({ type: "stream-end", key: LIVE } satisfies ToWorker);
    }, []),
    /** Drop the streamed preview once the recorded output replaces it. */
    clear: useCallback(() => {
      streaming.current = false;
      overlay.dropSource(LIVE);
      void overlay.rebuild();
    }, [overlay]),
  };

  return { overlay, version, status, stream };
}
