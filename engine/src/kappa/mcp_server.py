"""Kappa over MCP: the run lifecycle as tools an agent can call, under the same refusals.

An agent can start runs, draft and propose protocols, track and read results, but it
cannot make a number appear that the engine would withhold. Nothing is computed from
an unconfirmed protocol, and no tool here confirms one or applies an identity repair:
a name typed into a tool call proves nothing about who approved, so the approval gate
is the absence of the tool. The agent submits a protocol or proposes marks; a person
confirms or applies them with the ``kappa`` command or in the web workspace, and the
manifest records which. Results never carry a locked number's provisional value, nor
the behaviour bouts or verdicts it could be rebuilt from.

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

Workflow: create_run -> draft_protocol -> show the draft to a person -> \
submit_for_confirmation (the protocol they want, edited or not) -> a person confirms it \
themselves -> track -> get_results -> get_manifest. Optional: score_behaviour for \
behaviours in the protocol, propose_repair when results are locked by an identity doubt.

Rules you must keep:
- You draft and propose; a person confirms. No tool confirms a protocol or applies an \
identity repair. Give the person the command the tool returns (kappa confirm, kappa repair \
--from-proposal) or point them to the web workspace, then wait. list_runs shows the run as \
confirmed once they have; until then track and get_results refuse.
- Never present a protocol or marks as approved on your own judgement, and never guess marks: \
propose only what the person asked for or agreed to.
- Report only verified values as numbers. A locked result has value null: say it is withheld \
and why, and offer the fix (a person marking the animals at fix_frame). A refused result \
cannot be measured; give its reason. Never estimate a locked or refused value.
"""

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


def _load(runner: Runner, run_id: str):
    return runner.load(public.check_run_id(run_id))


def _missing(protocol: dict[str, Any]) -> list[str]:
    return [] if protocol.get("arena", {}).get("corners_px") is not None else ["arena.corners_px"]


class AnchorIn(BaseModel):
    """One animal marked on the repair frame, as a person asked for it to be marked."""

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
    quantities Kappa won't measure, with the reason). If they want changes, pass the
    edited protocol to submit_for_confirmation. Either way only the person can
    confirm it. The draft is also written to protocol.draft.json in the run folder.
    """
    with _refusals():
        runner = runner_factory()
        _load(runner, run_id)
        state = runner.draft(run_id, question)
        draft = state.draft or {}
        return {
            "run_id": run_id,
            "confirmed": False,
            "draft": draft,
            "draft_sha256": public.protocol_hash(draft),
            "assumptions": state.assumptions,
            "refused": draft.get("refused", []),
            "missing": _missing(draft),
            "draft_file": str(runner.draft_path(run_id)),
            "next": "show the draft, assumptions and refusals to a person; submit_for_confirmation "
            "with any edits they want, then they confirm it themselves",
        }


@mcp.tool()
def submit_for_confirmation(run_id: str, protocol_json: str | dict[str, Any]) -> dict[str, Any]:
    """Put a protocol in front of a person for confirmation. This does NOT confirm it.

    protocol_json is the protocol (a JSON string or object), e.g. the draft from
    draft_protocol with the edits the person asked for. It is checked and saved as
    the run's draft (protocol.draft.json in the run folder). A run that already has a
    confirmed protocol keeps measuring under it until a person confirms this one.
    Returns status "awaiting_person" and the command the person runs to confirm; they
    can also confirm it in the Kappa web workspace. Tell them, then wait: track and
    get_results refuse until they have. If arena.corners_px is missing, the person
    marks the corners before confirming.
    """
    with _refusals():
        runner = runner_factory()
        _load(runner, run_id)
        try:
            protocol = json.loads(protocol_json) if isinstance(protocol_json, str) else protocol_json
        except json.JSONDecodeError as error:
            raise RunError(f"protocol_json is not valid JSON: {error}") from error
        state = runner.propose(run_id, protocol, source="mcp")
        draft = state.draft or {}
        command = f'kappa confirm {run_id} --sha256 {public.protocol_hash(draft)[:16]} --by "<your name>"'
        return {
            "run_id": run_id,
            "status": "awaiting_person",
            "draft_sha256": public.protocol_hash(draft),
            "draft_file": str(runner.draft_path(run_id)),
            "missing": _missing(draft),
            "confirmed_protocol_sha256": public.protocol_hash(state.protocol) if state.confirmed_by else None,
            "command": command,
            "next": f"ask the person to review the protocol and confirm it themselves, by running `{command}` "
            "on this machine or in the Kappa web workspace; call list_runs to see when they have",
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
    (propose_repair, then the person applies it). refused: value is null and the
    reason says why it cannot be measured at all. Report only verified values as
    numbers; never estimate a locked or refused one. Fails if the protocol is
    unconfirmed or the video is untracked.
    """
    with _refusals():
        runner = runner_factory()
        return public.results_view(runner, _load(runner, run_id))


@mcp.tool()
def get_manifest(run_id: str) -> dict[str, Any]:
    """Build and return the run's manifest: everything its numbers depend on, by hash.

    Includes the video and protocol hashes, who confirmed the protocol and through
    which channel (cli or web), every identity repair with its channel, the models and
    their stored outputs, every integrity event, every result and a methods paragraph
    generated from the same fields. Locked and refused results carry no value here
    either, and behaviour bouts are listed only where every result for that behaviour
    and subject is verified. The full audit copy is manifest.json in the run folder.
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
    """List every run with its stage (created, drafted, confirmed, tracked), protocol hash, who confirmed it and how."""
    with _refusals():
        return {"runs": [public.run_row(s) for s in runner_factory().list()]}


@mcp.tool()
def propose_repair(run_id: str, frame: int, anchors: list[AnchorIn]) -> dict[str, Any]:
    """Propose marks that settle who is who on one frame. This does NOT apply them.

    Use it when get_results shows results locked by an identity event (contact or
    id_churn) and a person has told you where each animal is at that event's
    fix_frame: a point or a box per animal, under the subject number they say it is.
    Never guess the marks yourself. The marks are checked against the video and
    written to repair.proposal-<frame>.json in the run folder; nothing changes until
    the person applies them with the command returned here, after which SAM 3.1
    re-tracks from that frame and every check runs again. Returns status
    "awaiting_person" and that command.
    """
    with _refusals():
        runner = runner_factory()
        _load(runner, run_id)
        marks = [Anchor(a.subject, frame, box=a.box, point=a.point) for a in anchors]
        path = runner.propose_repair(run_id, marks, source="mcp")
        command = f'kappa repair {run_id} --from-proposal {frame} --sha256 {public.marks_hash(marks)[:16]} --by "<your name>"'
        return {
            "run_id": run_id,
            "frame": frame,
            "status": "awaiting_person",
            "proposal_file": str(path),
            "anchors": [a.model_dump(mode="json") for a in anchors],
            "command": command,
            "next": f"ask the person to check these marks on frame {frame} and apply them by running `{command}`; "
            "then get_results",
        }


@mcp.tool()
def score_behaviour(run_id: str, behaviour_id: str) -> dict[str, Any]:
    """Score one behaviour of the confirmed protocol (e.g. rearing) on the tracked video.

    Rearing is read frame by frame from the SAM 3.1 silhouettes by Kappa's rearing
    model (trained only on the pre-registered tuning videos). A scoring gap locks the
    windows it touches rather than counting
    as zero. Returns the gated results of the metrics that count this behaviour and
    how many candidates were reviewed. That count covers only subjects whose results
    for the behaviour are all verified; for the others (withheld_subjects) it would
    bound a withheld number, so it stays in the engine with their verdicts.
    """
    with _refusals():
        runner = runner_factory()
        _load(runner, run_id)
        state = runner.score(run_id, behaviour_id)
        record = state.behaviours[behaviour_id]
        ids = {m.id for m in state.confirmed.metrics if m.behaviour == behaviour_id}
        analysis = runner.analysis(state)
        view = public.results_view(runner, state, analysis)
        disclosed = public.disclosed_behaviours(analysis.results)
        reviewed = public.disclosed_verdicts(runner.store.get_json(record.verdicts_sha256), behaviour_id, disclosed)
        measured = {r.subject for r in analysis.results if r.params.get("behaviour") == behaviour_id}
        return {
            "run_id": run_id,
            "behaviour_id": behaviour_id,
            "bouts_found": len(reviewed),
            "withheld_subjects": sorted(s for s in measured if s is not None and (behaviour_id, s) not in disclosed),
            "reviews_failed": len(record.failed_spans),
            "results": [r for r in view["results"] if r["metric_id"] in ids],
            "next": "get_results",
        }


# -- resources --------------------------------------------------------------------


@mcp.resource("kappa://runs/{run_id}/manifest", mime_type="application/json")
def run_manifest(run_id: str) -> str:
    """The run's latest manifest as last built by get_manifest, locked values and their bouts withheld."""
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
