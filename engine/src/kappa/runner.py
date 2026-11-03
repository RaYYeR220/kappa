"""A measurement run, step by step, persisted after every step.

The steps are: ingest the video, draft a protocol from a question, have a person
confirm it, track, analyse, score behaviours, repair identities, review, and write
the manifest. Model calls happen only in ``draft``, ``track``, ``score``, ``repair``
and ``review``; their raw outputs go to the content-addressed store and the run
records their hashes, so ``analyse`` and ``manifest`` can be replayed offline.

Confirming and repairing record the channel the person used (``via``), so a manifest
says not only who approved but where; an agent can only ``propose`` a protocol or a
repair, for a person to apply.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from kappa.agents import adjudicator, protocol_agent, reviewer
from kappa.evidence import Manifest
from kappa.integrity import make_event
from kappa.model import BehaviourEvent, EventKind, IntegrityEvent, Severity, Span
from kappa.pipeline import Analysis, Repair, analyse
from kappa.protocol import AssayProtocol, BehaviourSpec
from kappa.repair import Anchor, prompt_objects
from kappa.sam.client import SAM_MODEL, ModelApi, track_concept, track_prompts
from kappa.store import Store
from kappa.video import VideoInfo, extract_frame, normalise

RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
DRAFT_FILE = "protocol.draft.json"

# Where a person confirmed or repaired: the kappa command, the web workspace, or a
# direct call into this module. There is deliberately no MCP channel: an agent can
# propose, never apply.
Via = Literal["cli", "web", "api"]


def default_home() -> Path:
    return Path(os.environ.get("KAPPA_HOME", Path.cwd() / "kappa-data"))


DETECTOR_DEFINITIONS = {
    "rearing_topdown": "The animal lifts both forepaws off the floor and stands up on its hind legs, either free-standing or with its forepaws against the wall.",
}
# How each behaviour looks from an overhead camera, so the model knows what to look for.
DETECTOR_HINTS = {
    "rearing_topdown": (
        "The clip is a zoomed crop around the animal from an overhead camera. Seen from directly above, an "
        "animal that rears up looks clearly shorter and rounder than when it walks on all fours, because its "
        "body points up toward the camera; the tail stays on the floor. Rearing is often against a wall."
    ),
}


def _crop_around(track, span: Span, width: int, height: int, scale: float = 3.5) -> tuple[int, int, int, int] | None:
    """A square around where the subject is during ``span``, about ``scale`` body lengths wide."""
    import numpy as np

    sl = slice(span.start, span.end)
    cx, cy, body = track.cx[sl], track.cy[sl], track.major_px[sl]
    if not np.isfinite(cx).any():
        return None
    side = int(min(max(float(np.nanmedian(body)) * scale, 96), min(width, height)))
    x = int(np.clip(np.nanmedian(cx) - side / 2, 0, width - side))
    y = int(np.clip(np.nanmedian(cy) - side / 2, 0, height - side))
    return (x, y, side - side % 2, side - side % 2)


def _wall_distance_cm(analysis: Analysis, protocol: AssayProtocol, subject: int, frame: int) -> float | None:
    track = analysis.tracks.get(subject)
    if track is None or track.x_cm is None or track.y_cm is None:
        return None
    x, y = float(track.x_cm[frame]), float(track.y_cm[frame])
    if math.isnan(x) or math.isnan(y):
        return None
    w, h = protocol.arena.width_cm, protocol.arena.height_cm
    if protocol.arena.shape == "circle":
        r = w / 2
        return r - ((x - r) ** 2 + (y - r) ** 2) ** 0.5
    return min(x, w - x, y, h - y)


def _matches_where(spec: BehaviourSpec, event: BehaviourEvent, analysis: Analysis, protocol: AssayProtocol) -> bool:
    if spec.where == "anywhere":
        return True
    distance = _wall_distance_cm(analysis, protocol, event.subject, (event.span.start + event.span.end - 1) // 2)
    if distance is None:
        return False
    near = distance <= spec.near_wall_cm
    return near if spec.where == "near_wall" else not near


class RepairRecord(BaseModel):
    frame: int
    raw_sha256: str
    by: str
    anchors: list[dict[str, Any]]
    via: Via | None = None  # None on repairs recorded before channels were kept


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
    draft_source: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    protocol: dict[str, Any] | None = None
    confirmed_by: str | None = None
    confirmed_via: Via | None = None
    file_id: str | None = None
    sam_main_sha256: str | None = None
    repairs: list[RepairRecord] = Field(default_factory=list)
    behaviours: dict[str, BehaviourRecord] = Field(default_factory=dict)
    review: dict[str, Any] | None = None
    draft_model: str | None = None
    arena_model: str | None = None
    segmenter_model: str | None = None  # what produced sam_main; None on runs from before backends
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
        self._llm = None
        self._sam = None

    @property
    def api(self) -> ModelApi:
        if self._api is None:
            self._api = ModelApi(store=self.store)
        return self._api

    @property
    def llm(self):
        """Where the Muse model calls go: Meta Model API, or Venice when ``KAPPA_LLM_BACKEND=venice``."""
        if self._llm is None:
            if os.environ.get("KAPPA_LLM_BACKEND") == "venice":
                from kappa.venice import VeniceApi

                self._llm = VeniceApi()
            else:
                self._llm = self.api
        return self._llm

    @property
    def sam(self):
        """Where segmentation runs: Meta Model API (default), or self-hosted SAM 3.1 on Modal
        when ``KAPPA_SAM_BACKEND=modal``. ``Runner.track`` and ``Runner.repair`` use it."""
        name = self._sam_backend()
        if name == "api":
            return self.api
        if name != "modal":
            raise RunError(f"unknown KAPPA_SAM_BACKEND {name!r}: use api or modal")
        if self._sam is None:
            from kappa.sam.modal_backend import ModalSam

            self._sam = ModalSam(store=self.store)
        return self._sam

    @staticmethod
    def _sam_backend() -> str:
        return os.environ.get("KAPPA_SAM_BACKEND", "api").strip().lower()

    def _sam_file(self, state: RunState) -> str:
        """The video's id on the selected segmentation backend, uploading it if needed."""
        sam = self.sam
        if self._sam_backend() == "api":  # Model API file ids are kept on the run
            if state.file_id is None:
                state.file_id = sam.upload(self.video_path(state))
                self.save(state)
            return state.file_id
        return sam.upload(self.video_path(state))  # content-addressed, remembered in the store

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

    def draft_path(self, run_id: str) -> Path:
        """Where a run's draft waits for a person to read, edit and confirm it."""
        return self.path(run_id) / DRAFT_FILE

    def repair_proposal_path(self, run_id: str, frame: int) -> Path:
        return self.path(run_id) / f"repair.proposal-{frame}.json"

    def _write_draft(self, state: RunState) -> Path:
        path = self.draft_path(state.run_id)
        path.write_text(json.dumps(state.draft, indent=2), encoding="utf-8")
        return path

    # -- steps -----------------------------------------------------------------

    def create(
        self, source: Path, start_s: float = 0.0, end_s: float | None = None, fps: float | None = None, run_id: str | None = None
    ) -> RunState:
        run_id = run_id or uuid.uuid4().hex[:12]
        if not RUN_ID.fullmatch(run_id):
            raise RunError("a run id is 1-64 letters, digits, '-' or '_'")
        if (self.path(run_id) / "run.json").exists():
            raise RunError(f"run {run_id} already exists")
        _, info, digest = normalise(source, self.home / "videos", start_s=start_s, end_s=end_s, fps=fps)
        state = RunState(
            run_id=run_id,
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
        """Draft a new protocol. Any confirmed protocol stays in force until a person confirms another."""
        state = self.load(run_id)
        info = state.info
        result = protocol_agent.draft_protocol(self.llm, question, self.keyframes(state), info.width, info.height)
        if result.protocol is None:
            raise RunError("the protocol could not be drafted: " + "; ".join(result.errors))
        state.question = question
        state.draft = result.protocol.model_dump(mode="json")
        state.draft_source = protocol_agent.SPARK_MODEL
        state.draft_model = protocol_agent.last_model()
        state.assumptions = result.assumptions
        self.save(state)
        self._write_draft(state)
        return state

    def propose(self, run_id: str, protocol: dict[str, Any], source: str) -> RunState:
        """Make ``protocol`` the run's draft, for a person to confirm. Nothing runs on it yet.

        This is all an agent can do with a protocol. It leaves an existing confirmation
        exactly as it was: the run keeps measuring under the protocol a person
        confirmed until a person confirms this one. Arena corners may still be missing;
        the person marks them before confirming.
        """
        state = self.load(run_id)
        state.draft = AssayProtocol.model_validate(protocol).model_dump(mode="json")
        state.draft_source = source
        self.save(state)
        self._write_draft(state)
        return state

    def confirm(self, run_id: str, protocol: dict[str, Any], by: str, via: Via = "api") -> RunState:
        """A person accepts a protocol (edited or not). Only a confirmed protocol can run.

        ``via`` is the channel they used; it goes into the manifest beside their name.
        """
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
        state.confirmed_via = via
        return self.save(state)

    def track(self, run_id: str, on_delta: Callable[[str], None] | None = None) -> RunState:
        state = self.load(run_id)
        protocol = state.confirmed
        file_id = self._sam_file(state)
        _, digest = track_concept(self.sam, file_id, protocol.subject.concept, on_delta)
        state.sam_main_sha256 = digest
        state.segmenter_model = getattr(self.sam, "model", SAM_MODEL)
        return self.save(state)

    def analysis(self, state: RunState, workers: int | None = None, analysed: Span | None = None) -> Analysis:
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
            analysed=analysed,
            workers=workers,
        )

    def score(self, run_id: str, behaviour_id: str, on_verdict: Callable[[adjudicator.Verdict], None] | None = None) -> RunState:
        """Score a behaviour, and every other behaviour of the protocol that shares its detector.

        Rearing is detected frame by frame from the SAM 3.1 silhouettes by Kappa's
        rearing model (see :mod:`kappa.behaviour_model`); each behaviour then keeps the
        bouts matching its ``where`` (supported near the wall, unsupported away from it),
        so the split behaviours always add up to the whole.
        """
        from kappa import behaviour_model

        state = self.load(run_id)
        protocol = state.confirmed
        spec = next((b for b in protocol.behaviours if b.id == behaviour_id), None)
        if spec is None:
            raise RunError(f"the protocol has no behaviour {behaviour_id}")
        if spec.detector != "rearing_topdown":
            raise RunError(f"no detector for {spec.detector} yet")
        siblings = [b for b in protocol.behaviours if b.detector == spec.detector]
        current = self.analysis(state)
        model = behaviour_model.load()
        model_sha = json.loads((behaviour_model.MODEL_DIR / "rearing_topdown.json").read_text(encoding="utf-8"))["sha256"]
        source = f"kappa-rearing/{model_sha[:12]} on sam-3.1 silhouettes"
        raw = self.store.get_text(state.sam_main_sha256)
        arena = (protocol.arena.width_cm, protocol.arena.height_cm)
        events: list[BehaviourEvent] = []
        rows = []
        for sid, track in current.tracks.items():
            features = behaviour_model.features_for(raw, track, arena, by_source=len(current.tracks) > 1)
            for span, p in model.bouts(model.probabilities(features), track.present, track.fps):
                events.append(BehaviourEvent("rearing", sid, span, p, source))
                rows.append({"subject": sid, "span": [span.start, span.end], "probability": round(p, 4), "model": source})
        verdicts_sha = self.store.put_json(rows)
        for sibling in siblings:
            kept = [e for e in events if _matches_where(sibling, e, current, protocol)]
            state.behaviours[sibling.id] = BehaviourRecord(
                verdicts_sha256=verdicts_sha,
                events=[{**asdict(e), "behaviour": sibling.id, "span": [e.span.start, e.span.end]} for e in kept],
                failed_spans=[],
            )
        return self.save(state)

    def _check_marks(self, state: RunState, anchors: Sequence[Anchor]) -> int:
        """The frame all ``anchors`` are on, after checking they fit the run's video."""
        state.confirmed  # noqa: B018 - raises if unconfirmed
        if state.sam_main_sha256 is None:
            raise RunError("the video has not been tracked yet")
        if not anchors:
            raise RunError("mark at least one subject")
        prompt_objects(anchors)  # one frame, each subject once
        info = state.info
        frame = anchors[0].frame
        if not 0 <= frame < info.n_frames:
            raise RunError(f"frame {frame} is outside the video (0-{info.n_frames - 1})")
        for a in anchors:
            xs = [a.point[0]] if a.point else []
            ys = [a.point[1]] if a.point else []
            if a.box:
                xs += [a.box[0], a.box[2]]
                ys += [a.box[1], a.box[3]]
            if not all(0 <= x <= info.width for x in xs) or not all(0 <= y <= info.height for y in ys):
                raise RunError(f"the mark for subject {a.subject} is outside the {info.width}x{info.height} frame")
        return frame

    def propose_repair(self, run_id: str, anchors: Sequence[Anchor], source: str) -> Path:
        """Write marks for a person to review and apply with ``kappa repair --from-proposal``.

        This is all an agent can do about an identity doubt: the marks change nothing
        until a person applies them, and the repair is then recorded under their name.
        """
        state = self.load(run_id)
        frame = self._check_marks(state, anchors)
        path = self.repair_proposal_path(run_id, frame)
        body = {
            "run_id": run_id,
            "frame": frame,
            "source": source,
            "proposed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "anchors": [asdict(a) for a in anchors],
        }
        path.write_text(json.dumps(body, indent=2), encoding="utf-8")
        return path

    def repair_proposal(self, run_id: str, frame: int) -> list[Anchor]:
        """The marks proposed for ``frame``, as anchors a person can apply."""
        self.load(run_id)
        path = self.repair_proposal_path(run_id, frame)
        if not path.exists():
            raise RunError(f"run {run_id} has no repair proposal for frame {frame}")
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            anchors = [
                Anchor(
                    subject=int(a["subject"]),
                    frame=frame,
                    box=tuple(int(v) for v in a["box"]) if a.get("box") is not None else None,
                    point=tuple(int(v) for v in a["point"]) if a.get("point") is not None else None,
                )
                for a in body["anchors"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise RunError(f"{path.name} is not a valid repair proposal: {error}") from error
        if body.get("frame") != frame:
            raise RunError(f"{path.name} is for frame {body.get('frame')}, not {frame}")
        return anchors

    def repair(self, run_id: str, anchors: Sequence[Anchor], by: str, via: Via = "api") -> RunState:
        """A person settles who is who on one frame; ``via`` is the channel they used."""
        if not by.strip():
            raise RunError("a repair needs the name of the person making it")
        state = self.load(run_id)
        frame = self._check_marks(state, anchors)
        info = state.info
        _, digest = track_prompts(
            self.sam, self._sam_file(state), (info.width, info.height), prompt_objects(anchors), "forward"
        )
        state.repairs = [r for r in state.repairs if r.frame != frame] + [
            RepairRecord(frame=frame, raw_sha256=digest, by=by.strip(), anchors=[asdict(a) for a in anchors], via=via)
        ]
        return self.save(state)

    def manifest(self, run_id: str, review: bool = False) -> dict[str, Any]:
        state = self.load(run_id)
        current = self.analysis(state)
        artifacts = {"sam_main": state.sam_main_sha256 or ""}
        artifacts |= {f"sam_repair_{r.frame}": r.raw_sha256 for r in state.repairs}
        artifacts |= {f"verdicts_{b}": rec.verdicts_sha256 for b, rec in state.behaviours.items()}
        models = {"segmenter": state.segmenter_model or SAM_MODEL}
        if state.draft is not None and state.draft_model:
            models["protocol"] = state.draft_model
        if state.arena_model:
            models["arena"] = state.arena_model
        for record in state.behaviours.values():
            used = {v.get("model") for v in self.store.get_json(record.verdicts_sha256) if v.get("model")}
            if used:
                models["behaviour"] = ", ".join(sorted(used))
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
            confirmed_via=state.confirmed_via,
            repairs=[r.model_dump(mode="json", exclude={"raw_sha256"}) for r in state.repairs],
            models=models,
            artifacts=artifacts,
            events=current.events,
            results=current.results,
            behaviour_events=behaviour_events,
            review=[state.review] if state.review else [],
        ).to_dict()
        if review:
            state.review = reviewer.review(self.llm, manifest)
            manifest["review"] = [state.review]
            manifest["models"]["reviewer"] = state.review.get("model", reviewer.WONDER_MODEL)
        # The stored manifest and runs/<id>/manifest.json are the operator's audit record:
        # complete, with every provisional value and every behaviour bout. Nothing sends
        # them out as they are; every surface passes them through public.redact_manifest.
        digest = self.store.put_json(manifest)
        state.manifest_sha256 = digest
        self.save(state)
        (self.path(run_id) / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest
