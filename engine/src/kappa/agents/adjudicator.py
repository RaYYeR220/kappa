"""Decide whether a candidate stretch of video really shows a behaviour.

Cheap kinematic rules propose candidates from the tracks; Muse Spark watches each
candidate clip and rules on it against the protocol's operational definition. The
model only confirms or rejects what the tracks proposed and must locate the bout in
the clip, so every accepted event is tied to frames a person can replay.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kappa.agents.protocol_agent import SPARK_MODEL, output_text
from kappa.model import BehaviourEvent, Span
from kappa.protocol import BehaviourSpec
from kappa.sam.client import ModelApi

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "confidence": {"type": "number"},
        "onset_s": {"type": ["number", "null"]},
        "offset_s": {"type": ["number", "null"]},
        "rationale": {"type": "string"},
    },
    "required": ["present", "confidence", "onset_s", "offset_s", "rationale"],
}


@dataclass(frozen=True)
class Candidate:
    subject: int
    span: Span
    clip: Path
    clip_start: int
    hint: str = ""


@dataclass(frozen=True)
class Verdict:
    candidate: Candidate
    present: bool
    confidence: float
    span: Span | None
    rationale: str
    error: str | None = None
    model: str = SPARK_MODEL


def _prompt(behaviour: BehaviourSpec, candidate: Candidate, fps: float) -> str:
    duration = candidate.span.length / fps
    return (
        f"This {duration:.1f}-second clip is from a top-down recording of a behavioural experiment. "
        f"{candidate.hint}\n\n"
        f"Behaviour: {behaviour.label}.\nOperational definition: {behaviour.definition}\n\n"
        "Decide whether the subject performs this behaviour in the clip according to the definition. "
        "If it does, give the onset and offset in seconds from the start of the clip. "
        "If the view really does not let you tell, answer present=false and say why. "
        "confidence is your probability (0 to 1) that the behaviour is present."
    )


def judge(api: ModelApi, behaviour: BehaviourSpec, candidate: Candidate, fps: float, effort: str = "medium") -> Verdict:
    data = base64.b64encode(candidate.clip.read_bytes()).decode()
    body = {
        "model": SPARK_MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _prompt(behaviour, candidate, fps)},
                    {"type": "input_video", "video_url": f"data:video/mp4;base64,{data}"},
                ],
            }
        ],
        "reasoning": {"effort": effort},
        "text": {"format": {"type": "json_schema", "name": "verdict", "schema": VERDICT_SCHEMA}},
        "max_output_tokens": 8000,
    }
    try:
        response = api.create(body)
        result = json.loads(output_text(response))
    except Exception as error:  # noqa: BLE001 - a failed call is a missing verdict, never a "no"
        return Verdict(candidate, False, 0.0, None, "", error=str(error))
    model = response.get("model", SPARK_MODEL)
    span = None
    if result["present"] and result["onset_s"] is not None and result["offset_s"] is not None:
        start = candidate.clip_start + round(max(result["onset_s"], 0.0) * fps)
        end = candidate.clip_start + round(max(result["offset_s"], 0.0) * fps)
        start = min(max(start, candidate.span.start), candidate.span.end - 1)
        end = min(max(end, start + 1), candidate.span.end)
        span = Span(start, end)
    elif result["present"]:
        span = candidate.span
    confidence = min(max(float(result["confidence"]), 0.0), 1.0)
    return Verdict(candidate, bool(result["present"]), confidence, span, result["rationale"], model=model)


def adjudicate(
    api: ModelApi,
    behaviour: BehaviourSpec,
    candidates: Sequence[Candidate],
    fps: float,
    min_confidence: float = 0.5,
    workers: int = 8,
    on_verdict: Callable[[Verdict], None] | None = None,
) -> tuple[list[BehaviourEvent], list[Verdict]]:
    """Judge every candidate; accepted ones become events, all verdicts are returned for audit.

    Calls that fail are returned with ``error`` set; the caller must treat the
    behaviour count as incomplete (locked), not as zero.
    """
    def run(candidate: Candidate) -> Verdict:
        verdict = judge(api, behaviour, candidate, fps)
        if on_verdict is not None:
            on_verdict(verdict)
        return verdict

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(run, candidates))
    events = [
        BehaviourEvent(
            behaviour=behaviour.id,
            subject=v.candidate.subject,
            span=v.span,
            confidence=v.confidence,
            source=v.model,
            rationale=v.rationale,
        )
        for v in verdicts
        if v.error is None and v.present and v.span is not None and v.confidence >= min_confidence
    ]
    return events, verdicts
