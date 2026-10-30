"""Draft an assay protocol from a plain-English question and a few keyframes.

Muse Spark reads the question and the frames and proposes a protocol built only
from Kappa's metric library. Anything it can't express with that library goes into
``refused`` with a reason instead of being approximated. The draft is never run as
is: a person confirms it first, and only the confirmed protocol's hash can start a run.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kappa.protocol import AssayProtocol
from kappa.sam.client import ModelApi

SPARK_MODEL = "muse-spark-1.3"

LIBRARY = """\
Metric kinds (each reported per subject unless noted; add bin_s to also report per time bin):
- distance: path length in cm. mean_speed: cm/s over the analysed window.
- time_in_zone, fraction_time_in_zone, zone_entries, latency_to_zone: need a zone id.
- immobility_time, immobility_bouts: speed below immobile_below_cm_s (default 2.0) for at least immobile_min_s (default 1.0).
- mean_inter_subject_distance: group metric, needs two or more subjects.
- behaviour_count, behaviour_duration: need a behaviour id.
Zone kinds: center_fraction (centred region covering `fraction` of the floor area), periphery_band (within `band_cm` of the wall), rect_cm ([x0,y0,x1,y1] in arena cm, origin top-left).
Entry rule: centroid, or body_fraction (>= body_fraction of the body inside; the automated stand-in for 'all four paws').
Behaviour detectors: rearing_topdown (rearing seen from above; set near_wall_cm to split supported vs unsupported), grooming_topdown.
Integrity: max_speed_cm_s is the fastest plausible movement of the subject between frames."""

INSTRUCTIONS = f"""\
You design behavioural assays for a measuring instrument that analyses top-down experiment video.
Read the researcher's question and the frames, then propose a protocol using ONLY this library:

{LIBRARY}

Rules:
- Every quantity the researcher asks for must become a metric, or an entry in `refused` with a one-sentence reason (for example, something not observable from a top-down camera, or not in the library). Never approximate an unsupported quantity with a different one.
- subject_concept is a short concrete noun phrase for the segmenter ("mouse", "zebrafish", "fly"), no attributes unless needed to exclude look-alikes.
- If the arena size is not stated, use the standard size for the paradigm and record that in `assumptions`. Every value you chose rather than read from the question goes in `assumptions`, phrased so a researcher can confirm or correct it.
- Use the conventional definitions of the paradigm (e.g. open field centre zone, elevated plus maze arms) and say which convention you used in `notes`."""

DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "subject_concept": {"type": "string"},
        "subject_label": {"type": "string"},
        "expected_count": {"type": "integer"},
        "arena_shape": {"type": "string", "enum": ["rect", "circle"]},
        "arena_width_cm": {"type": "number"},
        "arena_height_cm": {"type": "number"},
        "zones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["center_fraction", "periphery_band", "rect_cm"]},
                    "fraction": {"type": ["number", "null"]},
                    "band_cm": {"type": ["number", "null"]},
                    "rect_cm": {"type": ["array", "null"], "items": {"type": "number"}},
                },
                "required": ["id", "kind", "fraction", "band_cm", "rect_cm"],
            },
        },
        "entry_mode": {"type": "string", "enum": ["centroid", "body_fraction"]},
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string"},
                    "zone": {"type": ["string", "null"]},
                    "behaviour": {"type": ["string", "null"]},
                    "bin_s": {"type": ["number", "null"]},
                },
                "required": ["id", "kind", "zone", "behaviour", "bin_s"],
            },
        },
        "behaviours": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "definition": {"type": "string"},
                    "detector": {"type": "string", "enum": ["rearing_topdown", "grooming_topdown"]},
                    "near_wall_cm": {"type": ["number", "null"]},
                },
                "required": ["id", "label", "definition", "detector", "near_wall_cm"],
            },
        },
        "max_speed_cm_s": {"type": "number"},
        "refused": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"request": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["request", "reason"],
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": [
        "title", "subject_concept", "subject_label", "expected_count", "arena_shape",
        "arena_width_cm", "arena_height_cm", "zones", "entry_mode", "metrics", "behaviours",
        "max_speed_cm_s", "refused", "assumptions", "notes",
    ],
}

CORNERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object_name": {"type": "string"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["object_name", "x", "y"],
            },
        }
    },
    "required": ["points"],
}

RECT_CORNERS = (
    "the top-left corner of the arena floor",
    "the top-right corner of the arena floor",
    "the bottom-right corner of the arena floor",
    "the bottom-left corner of the arena floor",
)
CIRCLE_RIM = (
    "the leftmost point of the arena floor's rim",
    "the topmost point of the arena floor's rim",
    "the rightmost point of the arena floor's rim",
    "the bottommost point of the arena floor's rim",
)


@dataclass
class ProtocolDraft:
    protocol: AssayProtocol | None
    assumptions: list[str]
    errors: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _image_part(path: Path) -> dict[str, str]:
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{data}"}


def output_text(response: dict[str, Any]) -> str:
    """The assistant's text from a Responses API result (reasoning items skipped)."""
    texts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    return "".join(texts)


def _structured(api: ModelApi, content: list[dict], schema: dict, name: str, effort: str, instructions: str | None = None) -> dict:
    body: dict[str, Any] = {
        "model": SPARK_MODEL,
        "input": [{"type": "message", "role": "user", "content": content}],
        "reasoning": {"effort": effort},
        "text": {"format": {"type": "json_schema", "name": name, "schema": schema}},
        "max_output_tokens": 16000,
    }
    if instructions:
        body["instructions"] = instructions
    return json.loads(output_text(api.create(body)))


def locate_arena(api: ModelApi, keyframe: Path, shape: str, width: int, height: int) -> tuple[tuple[float, float], ...] | None:
    """Ask Muse Spark to point at the arena's floor corners (perception grounding, 0–1000 grid)."""
    names = RECT_CORNERS if shape == "rect" else CIRCLE_RIM
    quoted = ", ".join(f'"{n}"' for n in names)
    prompt = (
        f"Point to each of the following in the image: {quoted}. For each, answer in the format "
        '{"object_name": "<name>", "x": <int>, "y": <int>}. The coordinates should be in the 0-1000 range. '
        "If you cannot find one, omit it."
    )
    result = _structured(api, [{"type": "input_text", "text": prompt}, _image_part(keyframe)], CORNERS_SCHEMA, "points", "medium")
    found = {p["object_name"]: p for p in result.get("points", [])}
    if any(n not in found for n in names):
        return None
    clamp = lambda v, hi: min(max(v, 0), hi - 1)  # noqa: E731
    return tuple(
        (clamp(found[n]["x"] * width / 1000, width), clamp(found[n]["y"] * height / 1000, height)) for n in names
    )


def _to_protocol(draft: dict, question: str, corners) -> AssayProtocol:
    behaviours = [
        {k: v for k, v in b.items() if v is not None} for b in draft["behaviours"]
    ]
    return AssayProtocol.model_validate(
        {
            "title": draft["title"],
            "question": question,
            "subject": {
                "concept": draft["subject_concept"],
                "expected_count": draft["expected_count"],
                "label": draft["subject_label"][:40] or "subject",
            },
            "arena": {
                "shape": draft["arena_shape"],
                "width_cm": draft["arena_width_cm"],
                "height_cm": draft["arena_height_cm"],
                "corners_px": corners,
            },
            "zones": [{k: v for k, v in z.items() if v is not None} for z in draft["zones"]],
            "entry_rule": {"mode": draft["entry_mode"]},
            "metrics": [{k: v for k, v in m.items() if v is not None} for m in draft["metrics"]],
            "behaviours": behaviours,
            "integrity": {"max_speed_cm_s": draft["max_speed_cm_s"]},
            "refused": draft["refused"],
            "notes": draft["notes"],
        }
    )


def draft_protocol(
    api: ModelApi, question: str, keyframes: list[Path], width: int, height: int, effort: str = "high"
) -> ProtocolDraft:
    """Draft a protocol; on a schema violation, show the model the error once and retry."""
    content: list[dict] = [{"type": "input_text", "text": f"Researcher's question:\n{question}"}]
    content += [_image_part(p) for p in keyframes]
    draft = _structured(api, content, DRAFT_SCHEMA, "assay_protocol", effort, INSTRUCTIONS)
    corners = locate_arena(api, keyframes[0], draft["arena_shape"], width, height)
    errors: list[str] = []
    for attempt in range(2):
        try:
            protocol = _to_protocol(draft, question, corners)
            assumptions = list(draft["assumptions"])
            if corners is None:
                assumptions.append("The arena corners could not be located automatically; mark them on a frame.")
            return ProtocolDraft(protocol, assumptions, errors, {"draft": draft, "corners": corners})
        except ValidationError as error:
            errors.append(str(error))
            if attempt == 1:
                break
            fix = content + [
                {"type": "input_text", "text": f"Your previous draft was:\n{json.dumps(draft)}\n\nIt failed validation:\n{error}\n\nReturn a corrected draft."}
            ]
            draft = _structured(api, fix, DRAFT_SCHEMA, "assay_protocol", effort, INSTRUCTIONS)
    return ProtocolDraft(None, list(draft.get("assumptions", [])), errors, {"draft": draft, "corners": corners})
