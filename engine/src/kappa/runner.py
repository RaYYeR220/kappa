"""A measurement run, step by step, persisted after every step.

The steps are: ingest the video, draft a protocol from a question, have a person
confirm it, track, analyse, score behaviours, repair identities, review, and write
the manifest. Model calls happen only in ``draft``, ``track``, ``score``, ``repair``
and ``review``; their raw outputs go to the content-addressed store and the run
records their hashes, so ``analyse`` and ``manifest`` can be replayed offline.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kappa.agents import adjudicator, protocol_agent, reviewer
from kappa.behaviour import rearing_candidates
from kappa.evidence import Manifest
from kappa.integrity import make_event
from kappa.model import BehaviourEvent, EventKind, IntegrityEvent, Severity, Span
from kappa.pipeline import Analysis, Repair, analyse
from kappa.protocol import AssayProtocol
from kappa.repair import Anchor, prompt_objects
from kappa.sam.client import SAM_MODEL, ModelApi, track_concept, track_prompts
from kappa.store import Store
from kappa.video import VideoInfo, extract_clip, extract_frame, normalise


def default_home() -> Path:
    return Path(os.environ.get("KAPPA_HOME", Path.cwd() / "kappa-data"))


class RepairRecord(BaseModel):
    frame: int
    raw_sha256: str
    by: str
    anchors: list[dict[str, Any]]


class BehaviourRecord(BaseModel):
    verdicts_sha256: str
    events: list[dict[str, Any]]
    failed_spans: list[tuple[int, int, int]] = Field(default_factory=list)


class RunState(BaseModel):
    run_id: str
    created_at: str
    source_name: str
    video_sha256: str
    video: dict[str, Any]
    question: str | None = None
    draft: dict[str, Any] | None = None
    assumptions: list[str] = Field(default_factory=list)
    protocol: dict[str, Any] | None = None
    confirmed_by: str | None = None
    file_id: str | None = None
    sam_main_sha256: str | None = None
    repairs: list[RepairRecord] = Field(default_factory=list)
    behaviours: dict[str, BehaviourRecord] = Field(default_factory=dict)
    review: dict[str, Any] | None = None
    manifest_sha256: str | None = None

    @property
    def info(self) -> VideoInfo:
        return VideoInfo(**self.video)

    @property
    def confirmed(self) -> AssayProtocol:
        if self.protocol is None or self.confirmed_by is None:
            raise RunError("the protocol has not been confirmed by a person")
        return AssayProtocol.model_validate(self.protocol)


class RunError(RuntimeError):
    pass


class Runner:
    def __init__(self, home: Path | None = None, api: ModelApi | None = None) -> None:
        self.home = home or default_home()
        self.store = Store(self.home / "store")
        self.runs = self.home / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self._api = api

    @property
    def api(self) -> ModelApi:
        if self._api is None:
            self._api = ModelApi(store=self.store)
        return self._api

    # -- persistence -----------------------------------------------------------

    def path(self, run_id: str) -> Path:
        return self.runs / run_id

    def load(self, run_id: str) -> RunState:
        file = self.path(run_id) / "run.json"
        if not file.exists():
            raise RunError(f"no run {run_id}")
        return RunState.model_validate_json(file.read_text())

    def save(self, state: RunState) -> RunState:
        folder = self.path(state.run_id)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "run.json.tmp"
        tmp.write_text(state.model_dump_json(indent=2))
        tmp.replace(folder / "run.json")
        return state

    def list(self) -> list[RunState]:
        return [self.load(p.name) for p in sorted(self.runs.iterdir()) if (p / "run.json").exists()]

    def video_path(self, state: RunState) -> Path:
        return self.home / "videos" / f"{state.video_sha256}.mp4"

    # -- steps -----------------------------------------------------------------

    def create(
        self, source: Path, start_s: float = 0.0, end_s: float | None = None, fps: float | None = None, run_id: str | None = None
    ) -> RunState:
        _, info, digest = normalise(source, self.home / "videos", start_s=start_s, end_s=end_s, fps=fps)
        state = RunState(
            run_id=run_id or uuid.uuid4().hex[:12],
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            source_name=source.name,
            video_sha256=digest,
            video=info.to_dict(),
        )
        return self.save(state)

    def keyframes(self, state: RunState, count: int = 3) -> list[Path]:
        info = state.info
        folder = self.path(state.run_id) / "keyframes"
        folder.mkdir(exist_ok=True)
        frames = [int(info.n_frames * (i + 0.5) / count) for i in range(count)]
        return [extract_frame(self.video_path(state), f, info.fps, folder / f"{f:06d}.jpg") for f in frames]

    def draft(self, run_id: str, question: str) -> RunState:
        state = self.load(run_id)
        info = state.info
        result = protocol_agent.draft_protocol(self.api, question, self.keyframes(state), info.width, info.height)
        if result.protocol is None:
            raise RunError("the protocol could not be drafted: " + "; ".join(result.errors))
        state.question = question
        state.draft = result.protocol.model_dump(mode="json")
        state.assumptions = result.assumptions
        state.protocol = None
        state.confirmed_by = None
        return self.save(state)

    def confirm(self, run_id: str, protocol: dict[str, Any], by: str) -> RunState:
        """A person accepts a protocol (edited or not). Only a confirmed protocol can run."""
        if not by.strip():
            raise RunError("a confirmation needs the name of the person confirming")
        state = self.load(run_id)
        validated = AssayProtocol.model_validate(protocol)
        if validated.arena.corners_px is None:
            raise RunError("mark the arena corners before confirming")
        previous = state.protocol
        if previous is not None and previous["subject"]["concept"] != validated.subject.concept:
            state.sam_main_sha256 = None
            state.repairs = []
        if previous is not None and previous != validated.model_dump(mode="json"):
            state.behaviours = {}
        state.protocol = validated.model_dump(mode="json")
        state.confirmed_by = by.strip()
        return self.save(state)

    def track(self, run_id: str, on_delta: Callable[[str], None] | None = None) -> RunState:
        state = self.load(run_id)
        protocol = state.confirmed
        if state.file_id is None:
            state.file_id = self.api.upload(self.video_path(state))
            self.save(state)
        _, digest = track_concept(self.api, state.file_id, protocol.subject.concept, on_delta)
        state.sam_main_sha256 = digest
        return self.save(state)

    def analysis(self, state: RunState, workers: int | None = None) -> Analysis:
        if state.sam_main_sha256 is None:
            raise RunError("the video has not been tracked yet")
        protocol = state.confirmed
        repairs = [Repair(r.frame, self.store.get_text(r.raw_sha256), r.by) for r in state.repairs]
        events: list[BehaviourEvent] = []
        extra: list[IntegrityEvent] = []
        scored: list[str] = []
        for behaviour_id, record in state.behaviours.items():
            scored.append(behaviour_id)
            events += [
                BehaviourEvent(e["behaviour"], e["subject"], Span(*e["span"]), e["confidence"], e["source"], e["rationale"])
                for e in record.events
            ]
            for subject, start, end in record.failed_spans:
                extra.append(
                    make_event(EventKind.SCORING_INCOMPLETE, Severity.BLOCKING, (subject,), Span(start, end),
                               f"{behaviour_id}: the review of this candidate failed", start)
                )
        return analyse(
            protocol,
            state.info,
            self.store.get_text(state.sam_main_sha256),
            repairs=repairs,
            behaviour_events=events,
            scored_behaviours=scored,
            extra_events=extra,
            workers=workers,
        )

    def score(self, run_id: str, behaviour_id: str, on_verdict: Callable[[adjudicator.Verdict], None] | None = None) -> RunState:
        state = self.load(run_id)
        protocol = state.confirmed
        spec = next((b for b in protocol.behaviours if b.id == behaviour_id), None)
        if spec is None:
            raise RunError(f"the protocol has no behaviour {behaviour_id}")
        if spec.detector != "rearing_topdown":
            raise RunError(f"no candidate detector for {spec.detector} yet")
        current = self.analysis(state)
        info = state.info
        folder = self.path(run_id) / "clips"
        folder.mkdir(exist_ok=True)
        candidates = []
        for sid, track in current.tracks.items():
            for span in rearing_candidates(track):
                clip = folder / f"{behaviour_id}-{sid}-{span.start:06d}-{span.end:06d}.mp4"
                if not clip.exists():
                    extract_clip(self.video_path(state), span.start, span.end, info.fps, clip)
                candidates.append(adjudicator.Candidate(sid, span, clip, span.start))
        events, verdicts = adjudicator.adjudicate(self.api, spec, candidates, info.fps, on_verdict=on_verdict)
        record = [
            {
                "subject": v.candidate.subject,
                "candidate": [v.candidate.span.start, v.candidate.span.end],
                "present": v.present,
                "confidence": v.confidence,
                "span": [v.span.start, v.span.end] if v.span else None,
                "rationale": v.rationale,
                "error": v.error,
            }
            for v in verdicts
        ]
        state.behaviours[behaviour_id] = BehaviourRecord(
            verdicts_sha256=self.store.put_json(record),
            events=[{**asdict(e), "span": [e.span.start, e.span.end]} for e in events],
            failed_spans=[(v.candidate.subject, v.candidate.span.start, v.candidate.span.end) for v in verdicts if v.error],
        )
        return self.save(state)

    def repair(self, run_id: str, anchors: Sequence[Anchor], by: str) -> RunState:
        if not by.strip():
            raise RunError("a repair needs the name of the person making it")
        state = self.load(run_id)
        state.confirmed  # noqa: B018 - raises if unconfirmed
        if state.file_id is None:
            raise RunError("the video has not been tracked yet")
        info = state.info
        frame = anchors[0].frame
        _, digest = track_prompts(self.api, state.file_id, (info.width, info.height), prompt_objects(anchors), "forward")
        state.repairs = [r for r in state.repairs if r.frame != frame] + [
            RepairRecord(frame=frame, raw_sha256=digest, by=by.strip(), anchors=[asdict(a) for a in anchors])
        ]
        return self.save(state)

    def manifest(self, run_id: str, review: bool = False) -> dict[str, Any]:
        state = self.load(run_id)
        current = self.analysis(state)
        artifacts = {"sam_main": state.sam_main_sha256 or ""}
        artifacts |= {f"sam_repair_{r.frame}": r.raw_sha256 for r in state.repairs}
        artifacts |= {f"verdicts_{b}": rec.verdicts_sha256 for b, rec in state.behaviours.items()}
        models = {"segmenter": SAM_MODEL, "protocol": protocol_agent.SPARK_MODEL}
        if state.behaviours:
            models["adjudicator"] = protocol_agent.SPARK_MODEL
        behaviour_events: list[BehaviourEvent] = []
        for record in state.behaviours.values():
            behaviour_events += [
                BehaviourEvent(e["behaviour"], e["subject"], Span(*e["span"]), e["confidence"], e["source"], e["rationale"])
                for e in record.events
            ]
        manifest = Manifest(
            run_id=state.run_id,
            video_sha256=state.video_sha256,
            video=state.video,
            protocol=current.protocol,
            confirmed_by=state.confirmed_by or "",
            models=models,
            artifacts=artifacts,
            events=current.events,
            results=current.results,
            behaviour_events=behaviour_events,
            review=[state.review] if state.review else [],
        ).to_dict()
        if review:
            state.review = reviewer.review(self.api, manifest)
            manifest["review"] = [state.review]
            manifest["models"]["reviewer"] = reviewer.WONDER_MODEL
        digest = self.store.put_json(manifest)
        state.manifest_sha256 = digest
        self.save(state)
        (self.path(run_id) / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest
