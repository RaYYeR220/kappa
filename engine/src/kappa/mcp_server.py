"""Kappa over MCP: the run lifecycle as tools an agent can call, under the same refusals.

An agent can start runs, draft protocols, track and read results, but it cannot make
a number appear that the engine would withhold. Nothing is computed from an
unconfirmed protocol. Confirming a protocol and repairing an identity need the name of
the person who approved. Results never carry a locked number's provisional value.

Run it over stdio with ``kappa-mcp`` (needs the ``mcp`` extra).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

import anyio.from_thread
import anyio.to_thread
from pydantic import BaseModel, Field, model_validator

from kappa import library, public
from kappa.repair import Anchor
from kappa.runner import RunError, Runner

try:
    try:
        from mcp.server.mcpserver import Context
        from mcp.server.mcpserver import MCPServer as FastMCP  # mcp 2: FastMCP renamed MCPServer
        from mcp.server.mcpserver.exceptions import ResourceError, ToolError
    except ImportError:
        from mcp.server.fastmcp import Context, FastMCP  # mcp 1.x
        from mcp.server.fastmcp.exceptions import ResourceError, ToolError
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError("the Kappa MCP server needs the mcp extra: uv sync --extra mcp") from error

INSTRUCTIONS = """\
Kappa is a measuring instrument made from a camera: experiment video in, behavioural \
measurements out, each one verified, locked or refused.

Workflow: create_run -> draft_protocol -> show the draft to a person -> confirm_protocol \
(with their name) -> track -> get_results -> get_manifest. Optional: score_behaviour for \
behaviours in the protocol, repair_identity when results are locked by an identity doubt.

Rules you must keep:
- Never confirm a protocol or repair an identity on your own judgement. A person must have \
approved it; pass their name as confirmed_by.
- Report only verified values as numbers. A locked result has value null: say it is withheld \
and why, and offer the fix (a person marking the animals at fix_frame). A refused result \
cannot be measured; give its reason. Never estimate a locked or refused value.
"""

# Names that say an agent, not a person, approved.
NOT_A_PERSON = frozenset(
    {"agent", "ai", "assistant", "auto", "automatic", "bot", "human", "llm", "mcp", "model",
     "n/a", "na", "none", "nobody", "person", "system", "unknown", "user"}
)

mcp = FastMCP(name="kappa", instructions=INSTRUCTIONS)


@cache
def _shared_runner() -> Runner:
    return Runner()


# Tests swap this for a factory whose runner has a fake Model API.
runner_factory: Callable[[], Runner] = _shared_runner


@contextmanager
def _refusals() -> Iterator[None]:
    """Expected failures reach the agent as one plain sentence."""
    try:
        yield
    except ToolError:
        raise
    except public.EXPECTED_ERRORS as error:
        raise ToolError(public.plain_error(error)) from error


def _person(name: str, action: str) -> str:
    name = (name or "").strip()
    if not name:
        raise RunError(f"{action} needs the name of the person who approved it (confirmed_by)")
    if name.lower() in NOT_A_PERSON:
        raise RunError(f"{name!r} is not a person's name; ask the person who approved for their name")
    return name


def _load(runner: Runner, run_id: str):
    return runner.load(public.check_run_id(run_id))


class AnchorIn(BaseModel):
    """One animal a person marked on the repair frame."""

    subject: int = Field(ge=1, description="The subject number the person says this animal is (1, 2, ...).")
    point: tuple[int, int] | None = Field(default=None, description="[x, y] pixel on the animal.")
    box: tuple[int, int, int, int] | None = Field(default=None, description="[x1, y1, x2, y2] pixel box around the animal.")

    @model_validator(mode="after")
    def _marked(self) -> AnchorIn:
        if self.point is None and self.box is None:
            raise ValueError("each anchor needs a point or a box")
        return self


# -- tools ------------------------------------------------------------------------


@mcp.tool()
def create_run(video_path: str, start_s: float = 0.0, end_s: float | None = None) -> dict[str, Any]:
    """Start a measurement run from a video file on this machine.

    The video is re-encoded at a constant frame rate and hashed; every frame number
    Kappa reports refers to that normalised video. Trim with start_s and end_s
    (seconds). Returns the run id and the video's size, frame rate and frame count.
    Next: draft_protocol.
    """
    with _refusals():
        source = Path(video_path).expanduser()
        if not source.is_file():
            raise RunError(f"no video at {source}")
        state = runner_factory().create(source, start_s=start_s, end_s=end_s)
        return {
            "run_id": state.run_id,
            "video": state.video,
            "video_sha256": state.video_sha256,
            "next": "draft_protocol with the researcher's question",
        }


@mcp.tool()
def draft_protocol(run_id: str, question: str) -> dict[str, Any]:
    """Draft an assay protocol for a run from the researcher's question in plain English.

    Muse Spark reads the question and three keyframes and proposes a protocol from
    Kappa's metric library: subject, arena, zones, metrics, behaviours and integrity
    thresholds. The draft is NOT confirmed and nothing can be measured with it.
    Show the person the draft, every assumption and every refusal (asked-for
    quantities Kappa won't measure, with the reason). They may edit it. Then call
    confirm_protocol with the protocol they approved and their name. The draft is
    also written to protocol.draft.json in the run folder for them to edit.
    """
    with _refusals():
        runner = runner_factory()
        _load(runner, run_id)
        state = runner.draft(run_id, question)
        draft = state.draft or {}
        missing = [] if draft.get("arena", {}).get("corners_px") is not None else ["arena.corners_px"]
        return {
            "run_id": run_id,
            "confirmed": False,
            "draft": draft,
            "draft_sha256": public.protocol_hash(draft),
            "assumptions": state.assumptions,
            "refused": draft.get("refused", []),
            "missing": missing,
            "draft_file": str(public.write_draft(runner, state)),
            "next": "show the draft, assumptions and refusals to a person; after they approve, "
            "call confirm_protocol with the approved protocol and their name",
        }


@mcp.tool()
def confirm_protocol(run_id: str, protocol_json: str | dict[str, Any], confirmed_by: str) -> dict[str, Any]:
    """Record that a person approved this protocol for the run. Only a confirmed protocol can be measured.

    Call this ONLY after a human has read the protocol (with its assumptions and
    refusals) and approved it, as drafted or edited. confirmed_by is that person's
    name; it goes into the manifest and the methods paragraph. Never pass your own
    name or confirm on your own judgement. protocol_json is the approved protocol,
    as a JSON string or object (e.g. the draft from draft_protocol). The arena
    corners (arena.corners_px) must be set. Re-confirming a changed protocol clears
    results that depended on the old one. Returns the protocol's sha256.
    """
    with _refusals():
        runner = runner_factory()
        _load(runner, run_id)
        name = _person(confirmed_by, "confirming a protocol")
        try:
            protocol = json.loads(protocol_json) if isinstance(protocol_json, str) else protocol_json
        except json.JSONDecodeError as error:
            raise RunError(f"protocol_json is not valid JSON: {error}") from error
        state = runner.confirm(run_id, protocol, by=name)
        return {
            "run_id": run_id,
            "confirmed": True,
            "confirmed_by": state.confirmed_by,
            "protocol_sha256": public.protocol_hash(state.protocol),
            "next": "track",
        }


@mcp.tool()
async def track(run_id: str, ctx: Context = None) -> dict[str, Any]:  # type: ignore[assignment]
    """Track every subject through the run's video with SAM 3.1, under the confirmed protocol.

    Fails if no person has confirmed the protocol yet. Long videos take minutes;
    progress is reported as frame lines received. The segmenter's raw output is stored
    by hash, so tracking again is free. Returns the result counts (verified, locked,
    refused) and the open integrity events. Next: get_results.
    """
    with _refusals():
        runner = runner_factory()
        state = _load(runner, run_id)
        total = float(state.info.n_frames)
        lines = 0

        def on_delta(text: str) -> None:
            nonlocal lines
            before, lines = lines, lines + text.count("\n")
            if ctx is not None and lines // 25 != before // 25:
                anyio.from_thread.run(ctx.report_progress, min(lines, total), total, f"{lines} frame lines")

        state = await anyio.to_thread.run_sync(lambda: runner.track(run_id, on_delta=on_delta))
        view = await anyio.to_thread.run_sync(lambda: public.results_view(runner, state))
        return {
            "run_id": run_id,
            "segmenter_output_sha256": state.sam_main_sha256,
            "summary": view["summary"],
            "open_blockers": view["open_blockers"],
            "next": "get_results",
        }


@mcp.tool()
def get_results(run_id: str) -> dict[str, Any]:
    """Every result of the run, each verified, locked or refused.

    verified: value is the measurement (unit and uncertainty given). locked: value
    is null because an unresolved integrity event (for example two animals touching,
    after which who is who is uncertain) makes the number indefensible; reasons and
    blocked_by say which events, fix_frame is the frame where a person can settle it
    with repair_identity. refused: value is null and the reason says why it cannot be
    measured at all. Report only verified values as numbers; never estimate a locked
    or refused one. Fails if the protocol is unconfirmed or the video is untracked.
    """
    with _refusals():
        runner = runner_factory()
        return public.results_view(runner, _load(runner, run_id))


@mcp.tool()
def get_manifest(run_id: str) -> dict[str, Any]:
    """Build and return the run's manifest: everything its numbers depend on, by hash.

    Includes the video and protocol hashes, who confirmed the protocol, the models and
    their stored outputs, every integrity event, every result and a methods paragraph
    generated from the same fields. Locked and refused results carry no value here
    either. Also written to manifest.json in the run folder (the full audit copy).
    """
    with _refusals():
        runner = runner_factory()
        _load(runner, run_id)
        manifest = runner.manifest(run_id)
        state = runner.load(run_id)
        return public.redact_manifest(manifest) | {
            "manifest_sha256": state.manifest_sha256,
            "path": str(runner.path(run_id) / "manifest.json"),
        }


@mcp.tool()
def list_runs() -> dict[str, Any]:
    """List every run with its stage (created, drafted, confirmed, tracked), protocol hash and who confirmed it."""
    with _refusals():
        return {"runs": [public.run_row(s) for s in runner_factory().list()]}


@mcp.tool()
def repair_identity(run_id: str, frame: int, anchors: list[AnchorIn], confirmed_by: str) -> dict[str, Any]:
    """Settle who is who on one frame after a person marked each animal; SAM 3.1 re-tracks from there.

    Use it when get_results shows results locked by an identity event (contact or
    id_churn): a person looks at the video at that event's fix_frame and marks each
    animal with a point or a box under the subject number they say it is. Call this
    ONLY with marks a human made or approved, and pass their name as confirmed_by;
    never guess the marks yourself. Every check runs again on the repaired track, so
    a wrong click shows up as a new event. Returns the new counts and the events
    still open.
    """
    with _refusals():
        runner = runner_factory()
        _load(runner, run_id)
        name = _person(confirmed_by, "an identity repair")
        if not anchors:
            raise RunError("mark at least one subject")
        marks = [Anchor(a.subject, frame, box=a.box, point=a.point) for a in anchors]
        state = runner.repair(run_id, marks, by=name)
        view = public.results_view(runner, state)
        return {
            "run_id": run_id,
            "frame": frame,
            "confirmed_by": name,
            "summary": view["summary"],
            "open_blockers": view["open_blockers"],
            "next": "get_results",
        }


@mcp.tool()
def score_behaviour(run_id: str, behaviour_id: str) -> dict[str, Any]:
    """Score one behaviour of the confirmed protocol (e.g. rearing) on the tracked video.

    Kinematic rules propose candidate bouts from the tracks; Muse Spark reviews each
    candidate clip against the protocol's operational definition (one model call per
    candidate). A review that fails locks the windows it touches rather than counting
    as zero. Returns how many candidates were reviewed and the gated results of the
    metrics that count this behaviour.
    """
    with _refusals():
        runner = runner_factory()
        _load(runner, run_id)
        state = runner.score(run_id, behaviour_id)
        record = state.behaviours[behaviour_id]
        ids = {m.id for m in state.confirmed.metrics if m.behaviour == behaviour_id}
        view = public.results_view(runner, state)
        return {
            "run_id": run_id,
            "behaviour_id": behaviour_id,
            "candidates_reviewed": len(runner.store.get_json(record.verdicts_sha256)),
            "reviews_failed": len(record.failed_spans),
            "results": [r for r in view["results"] if r["metric_id"] in ids],
            "next": "get_results",
        }


# -- resources --------------------------------------------------------------------


@mcp.resource("kappa://runs/{run_id}/manifest", mime_type="application/json")
def run_manifest(run_id: str) -> str:
    """The run's latest manifest as last built by get_manifest, locked values withheld."""
    runner = runner_factory()
    try:
        state = _load(runner, run_id)
    except RunError as error:
        raise ResourceError(str(error)) from error
    if state.manifest_sha256 is None:
        raise ResourceError(f"run {run_id} has no manifest yet; call get_manifest to build it")
    manifest = runner.store.get_json(state.manifest_sha256)
    return json.dumps(public.redact_manifest(manifest) | {"manifest_sha256": state.manifest_sha256}, indent=2)


@mcp.resource("kappa://protocols/library", mime_type="application/json")
def protocol_library() -> str:
    """Ready protocols for common paradigms (open field, zebrafish novel tank, elevated plus maze).

    Their arena corners are null: a person marks them on a frame before confirming.
    """
    return json.dumps({"protocols": library.catalogue()}, indent=2)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
