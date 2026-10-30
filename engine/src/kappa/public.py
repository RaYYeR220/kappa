"""What a run shows outside the engine; shared by the command line, the MCP server,
the HTTP API and the static export.

Every surface reads results through here, so the refusal is the same on each: a
result carries a value only when it is verified. A locked result carries the events
that lock it and the frame where a person can settle them, never its provisional
number, and nothing it could be rebuilt from: not its evidence spans, not its
uncertainty, and not the behaviour bouts or adjudicator verdicts a locked count is
the tally of. The manifest file on disk keeps all of it for audit; a manifest sent
out through a surface has it removed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kappa.evidence import summary
from kappa.model import IntegrityEvent, MetricResult, Status
from kappa.pipeline import Analysis, NeedsCalibration
from kappa.protocol import AssayProtocol
from kappa.runner import RunError, Runner, RunState
from kappa.sam.client import ModelApiError

WITHHELD = (
    "Locked and refused results carry no value, evidence or uncertainty here, and behaviour "
    "bouts are listed only for a subject whose every result for that behaviour is verified: a "
    "locked number is not defensible until its integrity events are settled, and a locked count "
    "is just its bouts counted. The full manifest kept for audit is the file at `path`, sha256 "
    "`manifest_sha256`."
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# Errors a person can act on. Anything else is a bug and keeps its traceback.
EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    RunError,
    ValidationError,
    ModelApiError,
    NeedsCalibration,
    ValueError,
    KeyError,
    OSError,
    subprocess.CalledProcessError,
)


def check_run_id(run_id: str) -> str:
    if not _RUN_ID.match(run_id):
        raise RunError(f"{run_id!r} is not a run id (letters, digits, '.', '_' and '-', at most 64)")
    return run_id


def plain_error(error: BaseException) -> str:
    """One readable line for an expected error."""
    if isinstance(error, ValidationError):
        problems = [
            f"{'.'.join(str(p) for p in e['loc']) or 'protocol'}: {e['msg']}" for e in error.errors()[:6]
        ]
        more = len(error.errors()) - len(problems)
        tail = f" (and {more} more)" if more > 0 else ""
        return f"the protocol is not valid: {'; '.join(problems)}{tail}"
    if isinstance(error, subprocess.CalledProcessError):
        tool = Path(str(error.cmd[0])).stem if error.cmd else "a subprocess"
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr
        last = (stderr or "").strip().splitlines()[-1:] or [f"exit status {error.returncode}"]
        return f"{tool} could not process the video: {last[0]}"
    if isinstance(error, FileNotFoundError) and error.filename:
        return f"no such file: {error.filename}"
    if isinstance(error, OSError) and error.filename:
        return f"{error.strerror or error}: {error.filename}"
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error) or type(error).__name__


# -- runs ----------------------------------------------------------------------


def stage(state: RunState) -> str:
    if state.sam_main_sha256 is not None and state.confirmed_by is not None:
        return "tracked"
    if state.confirmed_by is not None:
        return "confirmed"
    if state.draft is not None:
        return "drafted"
    return "created"


def protocol_hash(protocol: Mapping[str, Any] | None) -> str | None:
    return AssayProtocol.model_validate(protocol).content_hash() if protocol else None


def marks_hash(anchors: Sequence[Any]) -> str:
    """The sha256 of a set of identity marks, so an approval names exactly what was approved."""
    rows = sorted(
        [a.subject, a.frame, list(a.box) if a.box else None, list(a.point) if a.point else None] for a in anchors
    )
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def check_pinned(expected: str | None, actual: str, what: str) -> None:
    """Refuse unless the person named the hash of the content they reviewed.

    The draft and proposal files sit on disk where an agent can rewrite them; pinning
    the approval to a hash means a swap between review and approval is refused.
    """
    if not expected:
        raise RunError(f"name the {what} you reviewed: add --sha256 {actual[:16]} (it is shown with the {what})")
    wanted = expected.strip().lower()
    if len(wanted) < 12 or not actual.startswith(wanted):
        raise RunError(f"the {what} on disk (sha256 {actual[:16]}) is not the one you named ({wanted[:16]}); review it again")


def run_row(state: RunState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "created_at": state.created_at,
        "video": state.source_name,
        "stage": stage(state),
        "protocol_sha256": protocol_hash(state.protocol),
        "confirmed_by": state.confirmed_by,
        "confirmed_via": state.confirmed_via,
    }


def read_draft(runner: Runner, state: RunState) -> dict[str, Any]:
    """The run's draft as a person left it: the edited file if there is one."""
    path = runner.draft_path(state.run_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if state.draft is not None:
        return state.draft
    raise RunError(f"run {state.run_id} has no draft yet; draft one or name a protocol file")


# -- results ---------------------------------------------------------------------


def _finite(values: Mapping[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in values.items() if v is not None and math.isfinite(v)}


def blocker(event: IntegrityEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "kind": event.kind.value,
        "subjects": list(event.subjects),
        "frames": [event.span.start, event.span.end],
        "detail": event.detail,
        "fix_frame": event.fix_frame,
    }


def public_result(result: MetricResult, events: Mapping[str, IntegrityEvent], fps: float) -> dict[str, Any]:
    """One result as it may leave the engine. A locked one has ``value: None`` and says why."""
    verified = result.status is Status.VERIFIED
    locked = result.status is Status.LOCKED
    blockers = [events[i] for i in result.reasons if i in events] if locked else []
    if locked:
        reasons = [f"{e.kind.value}, frames {e.span.start}-{e.span.end}: {e.detail}" for e in blockers]
        reasons += [i for i in result.reasons if i not in events]
    else:
        reasons = list(result.reasons)
    fixes = sorted(e.fix_frame for e in blockers if e.fix_frame is not None)
    span = result.bin
    return {
        "metric_id": result.metric_id,
        "subject": result.subject,
        "bin_frames": [span.start, span.end] if span else None,
        "bin_s": [round(span.start / fps, 3), round(span.end / fps, 3)] if span else None,
        "status": result.status.value,
        "value": result.value if verified else None,
        "unit": result.unit,
        "uncertainty": _finite(result.uncertainty) if verified else {},
        "reasons": reasons,
        "blocked_by": [e.id for e in blockers],
        "fix_frame": fixes[0] if fixes else None,
    }


def results_view(runner: Runner, state: RunState, analysis: Analysis | None = None) -> dict[str, Any]:
    """Every result of a run, gated: the only way a number leaves through a surface."""
    analysis = analysis or runner.analysis(state)
    events = {e.id: e for e in analysis.events}
    fps = state.info.fps
    return {
        "run_id": state.run_id,
        "protocol_sha256": analysis.protocol.content_hash(),
        "confirmed_by": state.confirmed_by,
        "confirmed_via": state.confirmed_via,
        "summary": summary(analysis.results),
        "results": [public_result(r, events, fps) for r in analysis.results],
        "open_blockers": [blocker(e) for e in analysis.events if e.is_open_blocker],
    }


def _behaviour_rows(results: Iterable[MetricResult | Mapping[str, Any]]) -> Iterator[tuple[str, int | None, bool]]:
    """(behaviour, subject, verified) for each behaviour result, from results or manifest rows."""
    for r in results:
        if isinstance(r, MetricResult):
            behaviour, subject, verified = r.params.get("behaviour"), r.subject, r.status is Status.VERIFIED
        else:
            params = r.get("params") or {}
            behaviour, subject = params.get("behaviour"), r.get("subject")
            verified = r.get("status") == Status.VERIFIED.value
        if behaviour is not None:
            yield behaviour, subject, verified


def disclosed_behaviours(results: Iterable[MetricResult | Mapping[str, Any]]) -> frozenset[tuple[str, int]]:
    """The (behaviour, subject) pairs whose bouts and verdicts may leave the engine.

    A pair qualifies only when it has results and every one of them is verified. A
    locked count is the number of accepted bouts, and the bouts are the verdicts that
    said yes, so showing either for a locked pair would show the withheld number. A
    behaviour no metric measures has no verified number behind its bouts, so they
    stay in as well. A result with no subject (the whole run refused) closes the
    behaviour for every subject.
    """
    verified: dict[tuple[str, int], bool] = {}
    closed: set[str] = set()
    for behaviour, subject, ok in _behaviour_rows(results):
        if subject is None:
            if not ok:
                closed.add(behaviour)
            continue
        verified[(behaviour, subject)] = verified.get((behaviour, subject), True) and ok
    return frozenset(pair for pair, ok in verified.items() if ok and pair[0] not in closed)


def disclosed_verdicts(
    rows: Iterable[Mapping[str, Any]], behaviour: str, disclosed: frozenset[tuple[str, int]]
) -> list[dict[str, Any]]:
    """The adjudicator's verdicts on ``behaviour`` for subjects whose results are all verified."""
    return [dict(r) for r in rows if (behaviour, r.get("subject")) in disclosed]


def redact_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """A manifest without anything a locked number could be rebuilt from.

    That is its provisional value, and also its evidence spans and uncertainty, since
    a locked time-in-zone is just the sum of its evidence spans, and the behaviour
    bouts of any (behaviour, subject) pair that is not verified throughout, since a
    locked count is just its bouts counted.
    """
    out = copy.deepcopy(dict(manifest))
    for result in out.get("results", []):
        result.pop("provisional", None)
        if result.get("status") != Status.VERIFIED.value:
            result["evidence"] = []
            result["uncertainty"] = {}
    disclosed = disclosed_behaviours(out.get("results", []))
    out["behaviour_events"] = [
        e for e in out.get("behaviour_events", []) if (e.get("behaviour"), e.get("subject")) in disclosed
    ]
    out["withheld"] = WITHHELD
    return out
