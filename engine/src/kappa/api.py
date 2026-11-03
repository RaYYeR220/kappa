"""HTTP API over the run lifecycle, for the web workspace.

Serve with ``uvicorn --factory kappa.api:create_app``.

Two modes. ``KAPPA_PUBLIC=1`` serves recorded runs read-only: nothing can call a
model, so a public deployment costs nothing and can't be abused. Otherwise every
state-changing request must carry ``Authorization: Bearer $KAPPA_OPERATOR_TOKEN``;
with no token configured, writes are refused unless ``KAPPA_INSECURE_DEV=1`` is set
explicitly for local development.

Confirmations and identity repairs made here are a person's acts in the web
workspace, and the run records them as such (``via="web"``). A protocol an agent
submitted over MCP shows up here as the run's draft, ready for that person to confirm.

Long model calls (tracking, scoring, review) run as background jobs. Tracking
streams the segmenter's raw output to the browser as it arrives, so the masks
appear frame by frame while SAM 3.1 is still working through the video.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from kappa.public import disclosed_behaviours, disclosed_verdicts, redact_manifest
from kappa.repair import Anchor
from kappa.runner import RunError, Runner, RunState
from kappa.views import event_view as _event
from kappa.views import result_view as _result
from kappa.views import rounded as _round

PUBLIC = os.environ.get("KAPPA_PUBLIC") == "1"


class Job:
    """A background step with a replayable event log (late subscribers see everything)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.events: list[dict[str, Any]] = []
        self.done = threading.Event()
        self.error: str | None = None
        self._cond = threading.Condition()

    def emit(self, event: dict[str, Any]) -> None:
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def finish(self, error: str | None = None) -> None:
        self.error = error
        self.emit({"type": "error", "message": error} if error else {"type": "done"})
        self.done.set()

    def follow(self, timeout_s: float = 15.0) -> Iterator[dict[str, Any] | None]:
        seen = 0
        while True:
            with self._cond:
                if seen >= len(self.events):
                    self._cond.wait(timeout_s)
                fresh, seen = self.events[seen:], len(self.events)
            if not fresh:
                yield None  # keep-alive
            for event in fresh:
                yield event
                if event["type"] in {"done", "error"}:
                    return


def _public_state(state: RunState) -> dict[str, Any]:
    """A run's record as any client may see it.

    Behaviour bouts are left out: a locked count is its bouts counted. The ones that
    may be shown come gated, in the manifest and from the verdicts endpoint.
    """
    data = state.model_dump(mode="json", exclude={"file_id": True, "behaviours": {"__all__": {"events"}}})
    data["confirmed"] = state.confirmed_by is not None
    data["tracked"] = state.sam_main_sha256 is not None
    return data


class DraftBody(BaseModel):
    question: str


class ConfirmBody(BaseModel):
    protocol: dict[str, Any]
    by: str


class AnchorBody(BaseModel):
    subject: int
    x: int | None = None
    y: int | None = None
    box: tuple[int, int, int, int] | None = None


class RepairBody(BaseModel):
    frame: int
    anchors: list[AnchorBody]
    by: str


def create_app(runner: Runner | None = None, public: bool = PUBLIC) -> FastAPI:
    app = FastAPI(title="Kappa", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("KAPPA_CORS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runner = runner or Runner()
    jobs: dict[tuple[str, str], Job] = {}
    cache: dict[tuple[str, str], Any] = {}

    def get_runner() -> Runner:
        return app.state.runner

    def operator(authorization: str | None = Header(default=None)) -> None:
        """Writes call paid models, so they fail closed: no token configured means no writes."""
        if public:
            raise HTTPException(403, "this deployment serves recorded runs only")
        token = os.environ.get("KAPPA_OPERATOR_TOKEN")
        if not token:
            if os.environ.get("KAPPA_INSECURE_DEV") == "1":
                return
            raise HTTPException(503, "KAPPA_OPERATOR_TOKEN is not configured; writes are disabled")
        if not authorization or not hmac.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(401, "operator token required")

    def load(run_id: str, r: Runner) -> RunState:
        try:
            return r.load(run_id)
        except RunError as error:
            raise HTTPException(404, str(error)) from error

    def analysis(state: RunState, r: Runner):
        key = (state.run_id, state.model_dump_json())
        if key not in cache:
            try:
                cache.clear()
                cache[key] = r.analysis(state)
            except RunError as error:
                raise HTTPException(409, str(error)) from error
        return cache[key]

    def start(run_id: str, name: str, work: Callable[[Job], None]) -> dict[str, str]:
        key = (run_id, name)
        if key in jobs and not jobs[key].done.is_set():
            raise HTTPException(409, f"{name} is already running")
        job = Job(name)
        jobs[key] = job

        def run() -> None:
            try:
                work(job)
                job.finish()
            except Exception as error:  # noqa: BLE001 - reported to the client as a job error
                job.finish(str(error))

        threading.Thread(target=run, daemon=True).start()
        return {"job": name, "stream": f"/runs/{run_id}/jobs/{name}"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "public": public}

    @app.get("/runs")
    def runs(r: Runner = Depends(get_runner)) -> list[dict[str, Any]]:
        return [_public_state(s) for s in r.list()]

    @app.post("/runs", dependencies=[Depends(operator)])
    def new_run(
        video: UploadFile = File(...),
        start_s: float = Form(0.0),
        end_s: float | None = Form(None),
        r: Runner = Depends(get_runner),
    ) -> dict[str, Any]:
        upload = r.home / "uploads" / f"{int(time.time() * 1000)}-{Path(video.filename or 'video.mp4').name}"
        upload.parent.mkdir(parents=True, exist_ok=True)
        with upload.open("wb") as out:
            while chunk := video.file.read(1 << 20):
                out.write(chunk)
        try:
            return _public_state(r.create(upload, start_s=start_s, end_s=end_s))
        except (RunError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/runs/{run_id}")
    def get_run(run_id: str, r: Runner = Depends(get_runner)) -> dict[str, Any]:
        return _public_state(load(run_id, r))

    @app.get("/runs/{run_id}/video")
    def video(run_id: str, r: Runner = Depends(get_runner)) -> FileResponse:
        return FileResponse(r.video_path(load(run_id, r)), media_type="video/mp4")

    @app.post("/runs/{run_id}/draft", dependencies=[Depends(operator)])
    def draft(run_id: str, body: DraftBody, r: Runner = Depends(get_runner)) -> dict[str, Any]:
        load(run_id, r)
        try:
            return _public_state(r.draft(run_id, body.question))
        except RunError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/runs/{run_id}/confirm", dependencies=[Depends(operator)])
    def confirm(run_id: str, body: ConfirmBody, r: Runner = Depends(get_runner)) -> dict[str, Any]:
        load(run_id, r)
        try:
            return _public_state(r.confirm(run_id, body.protocol, body.by, via="web"))
        except (RunError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/runs/{run_id}/track", dependencies=[Depends(operator)])
    def track(run_id: str, r: Runner = Depends(get_runner)) -> dict[str, str]:
        load(run_id, r).confirmed  # noqa: B018 - 409 below if unconfirmed
        def work(job: Job) -> None:
            r.track(run_id, on_delta=lambda delta: job.emit({"type": "sam", "delta": delta}))
        return start(run_id, "track", work)

    @app.post("/runs/{run_id}/score/{behaviour_id}", dependencies=[Depends(operator)])
    def score(run_id: str, behaviour_id: str, r: Runner = Depends(get_runner)) -> dict[str, str]:
        load(run_id, r)
        def work(job: Job) -> None:
            r.score(run_id, behaviour_id)
        return start(run_id, f"score-{behaviour_id}", work)

    @app.post("/runs/{run_id}/repair", dependencies=[Depends(operator)])
    def repair(run_id: str, body: RepairBody, r: Runner = Depends(get_runner)) -> dict[str, str]:
        load(run_id, r)
        anchors = [
            Anchor(a.subject, body.frame, box=a.box, point=(a.x, a.y) if a.x is not None and a.y is not None else None)
            for a in body.anchors
        ]
        def work(job: Job) -> None:
            r.repair(run_id, anchors, body.by, via="web")
        return start(run_id, f"repair-{body.frame}", work)

    @app.post("/runs/{run_id}/review", dependencies=[Depends(operator)])
    def review(run_id: str, r: Runner = Depends(get_runner)) -> dict[str, str]:
        load(run_id, r)
        def work(job: Job) -> None:
            r.manifest(run_id, review=True)
        return start(run_id, "review", work)

    @app.get("/runs/{run_id}/jobs/{name}", dependencies=[Depends(operator)])
    def follow(run_id: str, name: str) -> StreamingResponse:
        """Job progress, including error text from the model provider: operators only."""
        job = jobs.get((run_id, name))
        if job is None:
            raise HTTPException(404, f"no {name} job for this run")

        def stream() -> Iterator[str]:
            for event in job.follow():
                yield ": keep-alive\n\n" if event is None else f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/runs/{run_id}/results")
    def results(run_id: str, r: Runner = Depends(get_runner)) -> dict[str, Any]:
        state = load(run_id, r)
        a = analysis(state, r)
        return {
            "protocol_sha256": a.protocol.content_hash(),
            "analysed": [a.analysed.start, a.analysed.end],
            "results": [_result(x, {e.id: e for e in a.events}, state.info.fps) for x in a.results],
            "events": [_event(e) for e in a.events],
        }

    @app.get("/runs/{run_id}/tracks")
    def tracks(run_id: str, r: Runner = Depends(get_runner)) -> dict[str, Any]:
        state = load(run_id, r)
        a = analysis(state, r)
        out = {}
        for sid, t in a.tracks.items():
            out[str(sid)] = {
                "cx": _round(t.cx, 1),
                "cy": _round(t.cy, 1),
                "x_cm": _round(t.x_cm, 2),
                "y_cm": _round(t.y_cm, 2),
                "area_px": _round(t.area_px, 0),
                "interpolated": [int(v) for v in t.interpolated],
                "source": [int(v) for v in t.source],
            }
        return {"fps": state.info.fps, "n_frames": state.info.n_frames, "subjects": out}

    @app.get("/runs/{run_id}/sam")
    def sam_raw(run_id: str, r: Runner = Depends(get_runner)) -> PlainTextResponse:
        state = load(run_id, r)
        if state.sam_main_sha256 is None:
            raise HTTPException(409, "the video has not been tracked yet")
        return PlainTextResponse(r.store.get_text(state.sam_main_sha256))

    @app.get("/runs/{run_id}/manifest")
    def manifest(run_id: str, r: Runner = Depends(get_runner)) -> dict[str, Any]:
        state = load(run_id, r)
        if state.manifest_sha256 and r.store.has(state.manifest_sha256):
            return redact_manifest(r.store.get_json(state.manifest_sha256))
        if public:
            raise HTTPException(404, "no manifest recorded for this run")
        try:
            return redact_manifest(r.manifest(run_id))
        except RunError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/runs/{run_id}/verdicts/{behaviour_id}")
    def verdicts(run_id: str, behaviour_id: str, r: Runner = Depends(get_runner)) -> list[dict[str, Any]]:
        """The adjudicator's verdicts, only for subjects whose results for the behaviour are all verified."""
        state = load(run_id, r)
        record = state.behaviours.get(behaviour_id)
        if record is None:
            raise HTTPException(404, f"{behaviour_id} has not been scored")
        disclosed = disclosed_behaviours(analysis(state, r).results)
        return disclosed_verdicts(r.store.get_json(record.verdicts_sha256), behaviour_id, disclosed)

    return app

