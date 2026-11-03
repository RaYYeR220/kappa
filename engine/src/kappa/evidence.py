"""The run manifest: everything a reported number depends on, by hash.

A manifest is enough to audit a run without trusting Kappa: it names the exact
video, protocol, model outputs and code version, lists every integrity event and
every result with its status, and carries a methods paragraph generated from those
same fields (so the prose can't claim anything the run didn't do).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kappa import __version__
from kappa.model import BehaviourEvent, IntegrityEvent, MetricResult, Status
from kappa.protocol import AssayProtocol

MANIFEST_SCHEMA = "kappa.manifest/1"

# How each channel a person can approve through reads in a methods section.
CHANNELS = {"cli": "the kappa command line", "web": "the Kappa web workspace", "api": "the Kappa Python API"}


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@dataclass
class Manifest:
    run_id: str
    video_sha256: str
    video: Mapping[str, Any]
    protocol: AssayProtocol
    confirmed_by: str
    models: Mapping[str, str]
    artifacts: Mapping[str, str]
    events: Sequence[IntegrityEvent]
    results: Sequence[MetricResult]
    behaviour_events: Sequence[BehaviourEvent] = ()
    review: Sequence[Mapping[str, Any]] = ()
    confirmed_via: str | None = None
    repairs: Sequence[Mapping[str, Any]] = ()  # frame, by, via, anchors
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    engine_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "engine_version": self.engine_version,
            "video": {"sha256": self.video_sha256, **dict(self.video)},
            "protocol": {
                "sha256": self.protocol.content_hash(),
                "confirmed_by": confirmed_label(self.confirmed_by, self.confirmed_via),
                "confirmed_via": self.confirmed_via,
                "body": self.protocol.model_dump(mode="json"),
            },
            "models": dict(self.models),
            "artifacts": dict(self.artifacts),
            "repairs": _plain(list(self.repairs)),
            "integrity_events": _plain(list(self.events)),
            "behaviour_events": _plain(list(self.behaviour_events)),
            "results": _plain(list(self.results)),
            "review": _plain(list(self.review)),
            "summary": summary(self.results),
            "methods": methods_paragraph(self),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)


def confirmed_label(name: str, via: str | None) -> str:
    """``"Dr. Name (via cli)"``: who approved, and through which channel.

    A name alone can be typed by anyone, an agent included; the channel says which
    door the approval came through. MCP is not one of them: over MCP an agent can
    only propose.
    """
    return f"{name} (via {via})" if via else name


def _approver(name: str, via: str | None) -> str:
    return f"{name} via {CHANNELS.get(via, via)}" if via else name


def summary(results: Sequence[MetricResult]) -> dict[str, int]:
    counts = {s.value: 0 for s in Status}
    for r in results:
        counts[r.status.value] += 1
    return counts


def _zone_phrase(protocol: AssayProtocol, zone_id: str) -> str:
    zone = next(z for z in protocol.zones if z.id == zone_id)
    if zone.kind == "center_fraction":
        return f"a centre zone covering {zone.fraction:.0%} of the floor area"
    if zone.kind == "periphery_band":
        return f"a periphery band within {zone.band_cm:g} cm of the walls"
    x0, y0, x1, y1 = zone.rect_cm
    return f"a {x1 - x0:g} x {y1 - y0:g} cm zone"


def methods_paragraph(m: Manifest) -> str:
    """A methods-section paragraph built only from manifest fields."""
    p = m.protocol
    v = m.video
    arena = (
        f"a {p.arena.width_cm:g} x {p.arena.height_cm:g} cm arena"
        if p.arena.shape == "rect"
        else f"a circular arena {p.arena.width_cm:g} cm in diameter"
    )
    n = p.subject.expected_count
    subjects = f"{n} {p.subject.label}{'s' if n > 1 else ''}"
    zones = sorted({r.params.get("zone") for r in m.results if r.params.get("zone")})
    zone_text = ""
    if zones:
        zone_text = " Zones were defined as " + "; ".join(_zone_phrase(p, z) for z in zones) + "."
        if p.entry_rule.mode == "body_fraction":
            zone_text += f" A subject was counted inside a zone when at least {p.entry_rule.body_fraction:.0%} of its body area was inside it."
        else:
            zone_text += " A subject was counted inside a zone when its body centre was inside it."
    smoothing = (
        f" Trajectories were smoothed with a {p.smoothing.method} filter ({p.smoothing.window_s:g} s window)."
        if p.smoothing.method != "none"
        else ""
    )
    repairs = ""
    if m.repairs:
        settled = "; ".join(
            f"frame {r['frame']}, by {_approver(r['by'], r.get('via'))}" for r in sorted(m.repairs, key=lambda r: r["frame"])
        )
        repairs = (
            " Where the animals could not be told apart, a person marked each one on a frame and the segmenter "
            f"re-tracked forward from it ({settled})."
        )
    behaviours = ""
    if p.behaviours:
        detector = m.models.get("behaviour") or m.models.get("adjudicator")
        how = (
            f" Behaviour bouts were detected frame by frame with {detector}"
            if detector
            else " Behaviour bouts were detected from the segmentation"
        )
        behaviours = (
            how + ", split by the body centre's distance to the nearest wall where the protocol asks for it. Definitions: "
            + " ".join(f"{b.label}: {b.definition.rstrip('.')}." for b in p.behaviours)
        )
    counts = summary(m.results)
    gate = (
        f" Of {len(m.results)} reported values, {counts['verified']} were verified, {counts['locked']} were "
        f"withheld because the underlying track failed an integrity check, and {counts['refused']} were refused as not measurable."
    )
    return (
        f"Video ({v.get('width')}x{v.get('height')} px, {v.get('fps'):g} fps, {v.get('duration_s', 0) / 60:.1f} min; "
        f"sha256 {m.video_sha256[:12]}) of {subjects} in {arena} was analysed with Kappa {m.engine_version} "
        f"(protocol sha256 {p.content_hash()[:12]}, confirmed by {_approver(m.confirmed_by, m.confirmed_via)}). "
        f"Each {p.subject.label} was segmented and tracked with {m.models.get('segmenter', 'SAM 3.1')} from the text prompt "
        f"\"{p.subject.concept}\", and positions were converted to centimetres with a four-point homography of the arena floor."
        f"{repairs}{smoothing}{zone_text}{behaviours}{gate}"
    )
