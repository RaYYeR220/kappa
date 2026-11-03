"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { clock, frames as fmtFrames, modelName, plural, shortSha } from "@/lib/format";
import { API_BASE, follow, loadRun, readToken, startRepair, startReview, startTrack, writeToken } from "@/lib/live/api";
import { subjectColor, subjectTag } from "@/lib/palette";
import { Player } from "@/lib/player";
import type { EventView, ResultView, RunData, Status, TracksFile } from "@/lib/types";

import { Downloads, IntegrityList, MethodsCard, ProtocolCard, ReviewCard, VerdictsCard } from "./Details";
import { FixPanel, type MarkingState } from "./FixPanel";
import { binEdges, groupResults, isOpenBlocker, resultKey, subjectsOf } from "./model";
import { NumbersPanel } from "./NumbersPanel";
import { Timeline, TimelineLegend, type TimelineFocus } from "./Timeline";
import { Transport } from "./Transport";
import { useSam } from "./useSam";
import { type Layers, type Mark, VideoStage } from "./VideoStage";
import styles from "./Workspace.module.css";

// the operator token lives in this tab's sessionStorage only
const tokenListeners = new Set<() => void>();
const tokenStore = {
  subscribe(fn: () => void) {
    tokenListeners.add(fn);
    return () => {
      tokenListeners.delete(fn);
    };
  },
  get: () => readToken(),
  server: () => "",
  set(value: string) {
    writeToken(value);
    tokenListeners.forEach((fn) => fn());
  },
};

function summarise(results: ResultView[]): Record<Status, number> {
  const out: Record<Status, number> = { verified: 0, locked: 0, refused: 0 };
  for (const r of results) out[r.status] += 1;
  return out;
}

export function Workspace({ initial }: { initial: RunData }) {
  const live = initial.origin === "live" && !!API_BASE;
  const [run, setRun] = useState(initial);
  const [tracks, setTracks] = useState<TracksFile | null>(null);
  const [reloads, setReloads] = useState(0);
  const player = useMemo(() => new Player(run.video.fps, run.video.n_frames), [run.video.fps, run.video.n_frames]);
  const samUrl = run.urls.sam ? (live ? `${run.urls.sam}?v=${reloads}` : run.urls.sam) : null;
  const { overlay, version, status, stream } = useSam(samUrl, run.repairs, tracks, run.events);
  const [layers, setLayers] = useState<Layers>({ masks: true, trails: true, tags: true });
  const [selected, setSelected] = useState<ResultView | null>(null);
  const [fixFrame, setFixFrame] = useState<number | null>(null);
  const [marking, setMarking] = useState<MarkingState | null>(null);
  const [job, setJob] = useState<{ name: string; message: string | null; busy: boolean } | null>(null);
  const token = useSyncExternalStore(tokenStore.subscribe, tokenStore.get, tokenStore.server);
  const stageRef = useRef<HTMLDivElement>(null);

  const label = run.protocol?.subject.label ?? "subject";
  const fps = run.video.fps;
  const events = useMemo(() => new Map(run.events.map((e) => [e.id, e])), [run.events]);
  const groups = useMemo(() => groupResults(run), [run]);
  const bins = useMemo(() => binEdges(run), [run]);
  const trackIds = useMemo(() => (tracks ? Object.keys(tracks.subjects).map(Number) : []), [tracks]);
  const subjects = useMemo(() => subjectsOf(run, trackIds), [run, trackIds]);
  const summary = useMemo(() => summarise(run.results), [run.results]);

  useEffect(() => {
    if (!run.urls.tracks) return;
    const ctrl = new AbortController();
    fetch(run.urls.tracks, { signal: ctrl.signal, cache: live ? "no-store" : "default" })
      .then((r) => (r.ok ? (r.json() as Promise<TracksFile>) : null))
      .then((t) => setTracks(t))
      .catch(() => {});
    return () => ctrl.abort();
  }, [run.urls.tracks, live, reloads]);

  const focus: TimelineFocus | null = useMemo(() => {
    if (!selected) return null;
    if (selected.status === "verified")
      return { evidence: selected.evidence.map((e) => ({ span: e.span, subject: e.subject })), blockers: new Set(), window: selected.bin_frames };
    if (selected.status === "locked") return { evidence: [], blockers: new Set(selected.blocked_by), window: selected.bin_frames };
    return null;
  }, [selected]);

  const showVideo = useCallback(() => {
    const el = stageRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (r.top < 0 || r.bottom > window.innerHeight) el.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  }, []);

  const onSelect = useCallback(
    (r: ResultView | null) => {
      setSelected(r);
      if (r?.status === "verified" && r.evidence[0]) player.seek(r.evidence[0].span[0]);
    },
    [player],
  );

  const onGoTo = useCallback(
    (frame: number) => {
      player.pause();
      player.seek(frame);
      setFixFrame(frame);
      setMarking(null);
      showVideo();
    },
    [player, showVideo],
  );

  const onJumpEvent = useCallback(
    (e: EventView) => {
      player.pause();
      if (isOpenBlocker(e) && e.fix_frame != null) {
        onGoTo(e.fix_frame);
      } else {
        player.seek(e.span[0]);
        showVideo();
      }
    },
    [player, onGoTo, showVideo],
  );

  const onRef = useCallback(
    (id: string) => {
      const e = events.get(id);
      if (e) onJumpEvent(e);
    },
    [events, onJumpEvent],
  );

  const positions: Mark[] = useMemo(() => {
    if (fixFrame == null || !tracks) return [];
    return Object.entries(tracks.subjects)
      .map(([sid, t]) => ({ subject: Number(sid), x: Math.round(t.cx[fixFrame] ?? NaN), y: Math.round(t.cy[fixFrame] ?? NaN) }))
      .filter((m) => Number.isFinite(m.x) && Number.isFinite(m.y));
  }, [fixFrame, tracks]);

  const repair = fixFrame != null ? (run.repairs.find((r) => r.frame === fixFrame) ?? null) : null;
  const anchors = useMemo(
    () =>
      repair
        ? {
            frame: repair.frame,
            by: repair.by,
            marks: repair.anchors
              .map((a) => {
                const p = a.point ?? (a.box ? ([(a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2] as [number, number]) : null);
                return p ? { subject: a.subject, x: p[0], y: p[1] } : null;
              })
              .filter((m): m is Mark => m !== null),
          }
        : null,
    [repair],
  );

  // ---- live mode: tracking, repair and review run as jobs on the API ----

  const reload = useCallback(async () => {
    const next = await loadRun(run.id);
    setRun(next);
    setReloads((n) => n + 1);
    setSelected(null);
  }, [run.id]);

  const runJob = useCallback(
    async (name: string, start: () => Promise<{ stream: string }>, onEvent?: (e: { type: string; delta?: string }) => void) => {
      setJob({ name, message: null, busy: true });
      try {
        const { stream: path } = await start();
        for await (const e of follow(path, token)) {
          if (e.type === "error") throw new Error(e.message ?? "the job failed");
          onEvent?.(e);
        }
        await reload();
        setJob({ name, message: null, busy: false });
        return true;
      } catch (error) {
        setJob({ name, message: error instanceof Error ? error.message : String(error), busy: false });
        return false;
      }
    },
    [reload, token],
  );

  const onTrack = useCallback(async () => {
    stream.start();
    await runJob("track", () => startTrack(run.id, token), (e) => {
      if (e.type === "sam" && e.delta) stream.push(e.delta);
    });
    stream.end();
    stream.clear();
  }, [run.id, runJob, stream, token]);

  const onReview = useCallback(() => {
    void runJob("review", () => startReview(run.id, token));
  }, [run.id, runJob, token]);

  const markingProps =
    live && fixFrame != null
      ? {
          canWrite: !!token,
          marking,
          onStart: () => {
            setMarking({ active: positions[0]?.subject ?? null, marks: positions, by: "", busy: false, message: null });
            showVideo();
          },
          onActive: (s: number) => setMarking((m) => (m ? { ...m, active: s } : m)),
          onBy: (v: string) => setMarking((m) => (m ? { ...m, by: v } : m)),
          onCancel: () => setMarking(null),
          onSubmit: async () => {
            if (!marking || fixFrame == null) return;
            setMarking({ ...marking, busy: true, message: "SAM 3.1 is re-tracking forward from this frame." });
            const ok = await runJob("repair", () =>
              startRepair(run.id, token, { frame: fixFrame, by: marking.by.trim(), anchors: marking.marks.map((m) => ({ subject: m.subject, x: m.x, y: m.y })) }),
            );
            setMarking((m) => (m ? { ...m, busy: false, message: ok ? "Re-tracked. Every check ran again; the numbers below are current." : null } : m));
          },
        }
      : null;

  const stageMarking =
    marking && fixFrame != null
      ? {
          frame: fixFrame,
          active: marking.active,
          marks: marking.marks,
          onMark: (x: number, y: number) =>
            setMarking((m) => {
              if (!m || m.active == null) return m;
              const marks = [...m.marks.filter((k) => k.subject !== m.active), { subject: m.active, x, y }];
              const nextActive = subjects.find((s) => !marks.some((k) => k.subject === s)) ?? m.active;
              const hit = overlay.hitTest(fixFrame, x, y);
              return { ...m, marks, active: nextActive, message: hit ? null : "That point is not on a mask; SAM 3.1 will still try to segment what is there." };
            }),
        }
      : null;

  function onStageKey(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.target !== e.currentTarget) return;
    if (e.key === " " || e.key === "k") {
      e.preventDefault();
      player.toggle();
    } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const d = (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? Math.round(fps) : 1);
      player.step(d);
    }
  }

  const statusText =
    status.state === "loading"
      ? `Reading SAM 3.1 output, ${status.total ? Math.round((status.loaded / status.total) * 100) : 0}%`
      : status.state === "streaming"
        ? `SAM 3.1 is tracking: ${fmtFrames(status.masks)} masks so far`
        : status.state === "error"
          ? `Masks unavailable: ${status.message ?? "the segmenter output could not be read"}`
          : status.state === "ready"
            ? run.origin === "dev"
              ? `${fmtFrames(status.masks)} synthetic masks (development fixture)`
              : `${fmtFrames(status.masks)} masks from SAM 3.1`
            : run.tracked
              ? ""
              : "Not tracked yet";
  const unserved = run.repairs.find((r) => !r.sam);
  const statusLine =
    unserved && status.state === "ready"
      ? `${statusText}. From ${clock(unserved.frame, fps)} on the masks come from a re-track this server does not serve; positions are drawn there.`
      : statusText;

  const p = run.protocol;
  const models = run.manifest?.models ?? {};

  return (
    <div className={styles.workspace}>
      <header className={`wrap ${styles.head}`}>
        <nav aria-label="Breadcrumb" className={styles.crumbs}>
          <Link href="/runs">Recorded runs</Link>
          <span aria-hidden="true">/</span>
          <span className="mono" aria-current="page">
            {run.id}
          </span>
        </nav>
        {run.origin === "dev" && (
          <p className={styles.fixture} role="note">
            <strong>Development fixture.</strong> Synthetic masks: ellipses drawn from 3D-ZeF ground-truth boxes, not SAM 3.1
            output. It exercises the workspace; its numbers are not measurements.
          </p>
        )}
        <div className={styles.titleBlock}>
          <div className={styles.titleRow}>
            <h1 id="main">{p?.title ?? run.title}</h1>
            {run.question && <p className={styles.ask}>{run.question}</p>}
          </div>
          <dl className={styles.meta}>
            <div>
              <dt>Video</dt>
              <dd>
                {run.video.width} × {run.video.height}, {run.video.fps} fps
              </dd>
            </div>
            <div>
              <dt>Length</dt>
              <dd>
                {run.video.duration_s.toFixed(1)} s, {fmtFrames(run.video.n_frames)} frames
              </dd>
            </div>
            {p && (
              <div>
                <dt>Subjects</dt>
                <dd>
                  {p.subject.expected_count} {p.subject.expected_count === 1 ? label : plural(label)}
                </dd>
              </div>
            )}
            <div>
              <dt>Protocol</dt>
              <dd className="mono">{shortSha(run.protocolSha256)}</dd>
            </div>
            <div>
              <dt>Confirmed by</dt>
              <dd>{run.confirmedBy ?? "not yet"}</dd>
            </div>
            {Object.keys(models).length > 0 && (
              <div>
                <dt>Models</dt>
                <dd className={styles.wrap}>{[...new Set(Object.values(models).map(modelName))].join(", ")}</dd>
              </div>
            )}
          </dl>
        </div>
      </header>

      {live && (
        <LiveBar
          token={token}
          tracked={run.tracked}
          confirmed={!!run.confirmedBy}
          job={job}
          onTrack={() => void onTrack()}
        />
      )}

      <div className={`wrap ${styles.main}`}>
        <div className={styles.videoCol}>
          <div
            ref={stageRef}
            className={styles.well}
            tabIndex={0}
            role="group"
            aria-label="Video with masks. Space plays or pauses, arrow keys step one frame."
            onKeyDown={onStageKey}
          >
            <VideoStage
              src={run.urls.video}
              width={run.video.width}
              height={run.video.height}
              tracks={tracks}
              overlay={overlay}
              version={version}
              player={player}
              layers={status.state === "streaming" || marking ? { ...layers, tags: false, trails: false } : layers}
              label={label}
              events={run.events}
              focusSubject={selected?.subject ?? null}
              anchors={anchors}
              marking={stageMarking}
            >
              {run.origin === "dev" && <span className={styles.fixtureBadge}>Development fixture: synthetic masks</span>}
              {marking && fixFrame != null && (
                <div className={styles.markBar} role="group" aria-label={`Marking frame ${fixFrame}`}>
                  <span>
                    {marking.active != null ? `Click ${subjectTag(marking.active, label)} on the video` : "Choose a subject"}
                  </span>
                  {subjects.map((s) => (
                    <button
                      key={s}
                      type="button"
                      aria-pressed={marking.active === s}
                      onClick={() => setMarking((m) => (m ? { ...m, active: s } : m))}
                      style={{ background: subjectColor(s, "video") }}
                    >
                      {subjectTag(s, label)}
                    </button>
                  ))}
                </div>
              )}
            </VideoStage>
            <Transport player={player} layers={layers} onLayers={setLayers} status={statusLine} />
          </div>
          <Timeline player={player} subjects={subjects} label={label} tracks={tracks} events={run.events} bins={bins} focus={focus} />
          <TimelineLegend />
          {fixFrame != null && (
            <FixPanel
              runId={run.id}
              frame={fixFrame}
              fps={fps}
              label={label}
              subjects={subjects}
              blockers={run.events.filter((e) => e.fix_frame === fixFrame)}
              repair={repair}
              positions={positions}
              live={markingProps}
              onClose={() => {
                setFixFrame(null);
                setMarking(null);
              }}
            />
          )}
        </div>
        <aside className={styles.side} aria-label="Numbers">
          <NumbersPanel
            groups={groups}
            protocol={p}
            label={label}
            fps={fps}
            events={events}
            summary={summary}
            selected={selected ? resultKey(selected) : null}
            onSelect={onSelect}
            onGoTo={onGoTo}
            onEvidence={(f) => {
              player.seek(f);
              showVideo();
            }}
          />
        </aside>
        <div className={styles.checks}>
          <IntegrityList run={run} label={label} onJump={onJumpEvent} />
        </div>
      </div>

      <div className={`wrap ${styles.details}`}>
        <ProtocolCard run={run} />
        <div className={styles.stack}>
          <ReviewCard
            run={run}
            onRef={onRef}
            live={live ? { canWrite: !!token, busy: job?.name === "review" && job.busy, message: job?.name === "review" ? job.message : null, onRequest: onReview } : null}
          />
          <MethodsCard run={run} />
          <Downloads run={run} />
        </div>
        <VerdictsCard
          run={run}
          label={label}
          onJump={(f) => {
            player.seek(f);
            showVideo();
          }}
        />
      </div>
    </div>
  );
}

function LiveBar({
  token,
  tracked,
  confirmed,
  job,
  onTrack,
}: {
  token: string;
  tracked: boolean;
  confirmed: boolean;
  job: { name: string; message: string | null; busy: boolean } | null;
  onTrack: () => void;
}) {
  const [draft, setDraft] = useState("");
  return (
    <section className={`wrap ${styles.live}`} aria-label="Live workspace">
      <div className={styles.liveInner}>
        <p className={styles.liveWhere}>
          <span className={styles.liveDot} aria-hidden="true" />
          Live on <span className="mono">{API_BASE}</span>
        </p>
        {token ? (
          <p className={styles.liveToken}>
            Operator token set for this tab.
            <button type="button" className={styles.textButton} onClick={() => tokenStore.set("")}>
              Forget it
            </button>
          </p>
        ) : (
          <form
            className={styles.liveToken}
            onSubmit={(e) => {
              e.preventDefault();
              tokenStore.set(draft.trim());
              setDraft("");
            }}
          >
            <label>
              <span className="sr-only">Operator token</span>
              <input type="password" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Operator token" autoComplete="off" />
            </label>
            <button type="submit" className={styles.secondary} disabled={!draft.trim()}>
              Use token
            </button>
          </form>
        )}
        <div className={styles.liveActions}>
          <button
            type="button"
            className={styles.primary}
            disabled={!token || !confirmed || !!job?.busy}
            onClick={onTrack}
            title={confirmed ? undefined : "Confirm the protocol first: kappa confirm RUN --by NAME"}
          >
            {job?.name === "track" && job.busy ? "Tracking…" : tracked ? "Track again with SAM 3.1" : "Track with SAM 3.1"}
          </button>
        </div>
      </div>
      {job?.message && (
        <p className={styles.liveError} role="alert">
          {job.name}: {job.message}
        </p>
      )}
    </section>
  );
}
