"""The assay protocol: a complete, reviewable definition of what gets measured.

A protocol is drafted from a plain-English question, confirmed by a person, and
then frozen by its content hash. Every reported number names the protocol hash it
was computed under, so two labs running the same hash ran the same assay.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "kappa.protocol/1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SubjectSpec(_Strict):
    concept: str = Field(min_length=1, max_length=60, description="Noun phrase passed to the segmenter, e.g. 'mouse'.")
    expected_count: int = Field(ge=1, le=16)
    label: str = Field(default="subject", max_length=40)


class ArenaSpec(_Strict):
    """The physical arena and where it sits in the image.

    For ``rect`` arenas, ``corners_px`` lists the floor corners in image pixels in the
    order top-left, top-right, bottom-right, bottom-left (as seen in the image).
    For ``circle`` arenas, the four points are the floor's extreme left, top, right
    and bottom rim points. ``None`` until located on a keyframe.
    """

    shape: Literal["rect", "circle"]
    width_cm: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    corners_px: tuple[tuple[float, float], ...] | None = None

    @model_validator(mode="after")
    def _check(self) -> ArenaSpec:
        if self.corners_px is not None and len(self.corners_px) != 4:
            raise ValueError("corners_px needs exactly four points")
        if self.shape == "circle" and self.width_cm != self.height_cm:
            raise ValueError("a circular arena has equal width and height (its diameter)")
        return self


class ZoneSpec(_Strict):
    """A region of the arena floor.

    - ``center_fraction``: centred region with the arena's shape covering ``fraction`` of its area.
    - ``periphery_band``: everything within ``band_cm`` of the arena wall.
    - ``rect_cm``: axis-aligned rectangle ``[x0, y0, x1, y1]`` in arena centimetres.
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    kind: Literal["center_fraction", "periphery_band", "rect_cm"]
    fraction: float | None = Field(default=None, gt=0, lt=1)
    band_cm: float | None = Field(default=None, gt=0)
    rect_cm: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def _check(self) -> ZoneSpec:
        need = {"center_fraction": "fraction", "periphery_band": "band_cm", "rect_cm": "rect_cm"}[self.kind]
        if getattr(self, need) is None:
            raise ValueError(f"zone {self.id}: kind {self.kind} needs {need}")
        return self


class EntryRule(_Strict):
    """When a subject counts as inside a zone.

    ``centroid``: the body centre is inside. ``body_fraction``: at least ``body_fraction``
    of the body is inside (the automated stand-in for the 'all four paws' rule).
    A visit shorter than ``min_dwell_s`` is not an entry.
    """

    mode: Literal["centroid", "body_fraction"] = "centroid"
    body_fraction: float = Field(default=0.7, gt=0, le=1)
    min_dwell_s: float = Field(default=0.0, ge=0)


MetricKind = Literal[
    "distance",
    "mean_speed",
    "time_in_zone",
    "fraction_time_in_zone",
    "zone_entries",
    "latency_to_zone",
    "immobility_time",
    "immobility_bouts",
    "mean_inter_subject_distance",
    "behaviour_count",
    "behaviour_duration",
]


class MetricSpec(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    kind: MetricKind
    zone: str | None = None
    behaviour: str | None = None
    bin_s: float | None = Field(default=None, gt=0, description="Also report per time bin of this length.")
    immobile_below_cm_s: float = Field(default=2.0, gt=0)
    immobile_min_s: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> MetricSpec:
        zoned = {"time_in_zone", "fraction_time_in_zone", "zone_entries", "latency_to_zone"}
        if self.kind in zoned and self.zone is None:
            raise ValueError(f"metric {self.id}: {self.kind} needs a zone")
        if self.kind.startswith("behaviour_") and self.behaviour is None:
            raise ValueError(f"metric {self.id}: {self.kind} needs a behaviour")
        return self


class BehaviourSpec(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    label: str
    definition: str = Field(min_length=10, description="Operational definition, as it would appear in a methods section.")
    detector: Literal["rearing_topdown", "grooming_topdown"]
    adjudicate: bool = True
    near_wall_cm: float | None = Field(default=None, gt=0, description="Distance from the wall that separates supported from unsupported bouts.")
    where: Literal["anywhere", "near_wall", "away_from_wall"] = Field(
        default="anywhere",
        description="Which bouts count, by the body centre's distance to the nearest wall at mid-bout (e.g. supported rearing = near_wall).",
    )

    @model_validator(mode="after")
    def _check_where(self) -> BehaviourSpec:
        if self.where != "anywhere" and self.near_wall_cm is None:
            raise ValueError(f"behaviour {self.id}: where={self.where} needs near_wall_cm")
        return self


class IntegrityPolicy(_Strict):
    max_speed_cm_s: float = Field(default=150.0, gt=0, description="Faster than this between frames is physically implausible.")
    max_interpolated_gap_s: float = Field(default=0.4, ge=0)
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    low_confidence_min_s: float = Field(default=0.5, ge=0)
    contact_distance: float = Field(default=0.5, gt=0, description="Two subjects are in contact (their bodies touch or overlap, so the segmenter may confuse them) when their centres are closer than this many mean body lengths.")
    identity_margin: float = Field(default=2.0, gt=1, description="After a contact, identities count as resolved only if keeping them fits the motion this many times better than swapping them.")


class Smoothing(_Strict):
    method: Literal["none", "median", "savgol"] = "median"
    window_s: float = Field(default=0.2, ge=0)
    min_step_cm: float = Field(default=0.0, ge=0, description="Steps shorter than this are treated as jitter, not movement.")


class Sampling(_Strict):
    fps: float | None = Field(default=None, gt=0, description="Resample to this rate; None keeps the source rate.")
    start_s: float = Field(default=0.0, ge=0)
    end_s: float | None = Field(default=None, gt=0)


class Refusal(_Strict):
    request: str
    reason: str


class AssayProtocol(_Strict):
    schema_version: Literal["kappa.protocol/1"] = SCHEMA_VERSION
    title: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=1, max_length=2000)
    subject: SubjectSpec
    arena: ArenaSpec
    zones: tuple[ZoneSpec, ...] = ()
    entry_rule: EntryRule = EntryRule()
    metrics: tuple[MetricSpec, ...] = Field(min_length=1)
    behaviours: tuple[BehaviourSpec, ...] = ()
    integrity: IntegrityPolicy = IntegrityPolicy()
    smoothing: Smoothing = Smoothing()
    sampling: Sampling = Sampling()
    refused: tuple[Refusal, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def _check_refs(self) -> AssayProtocol:
        zone_ids = [z.id for z in self.zones]
        behaviour_ids = [b.id for b in self.behaviours]
        metric_ids = [m.id for m in self.metrics]
        for name, ids in (("zone", zone_ids), ("behaviour", behaviour_ids), ("metric", metric_ids)):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {name} id")
        for m in self.metrics:
            if m.zone is not None and m.zone not in zone_ids:
                raise ValueError(f"metric {m.id} refers to unknown zone {m.zone}")
            if m.behaviour is not None and m.behaviour not in behaviour_ids:
                raise ValueError(f"metric {m.id} refers to unknown behaviour {m.behaviour}")
            if m.kind == "mean_inter_subject_distance" and self.subject.expected_count < 2:
                raise ValueError(f"metric {m.id} needs at least two subjects")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
