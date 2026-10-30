"""Core data types shared by every stage of the pipeline.

Frames are integer indices into the normalized video. Spans are half-open
``[start, end)``. Pixel coordinates are in the media's displayed size; arena
coordinates are centimetres with the origin at the arena's top-left corner.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.end

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def intersect(self, other: Span) -> Span | None:
        start, end = max(self.start, other.start), min(self.end, other.end)
        return Span(start, end) if start < end else None


@dataclass(frozen=True)
class Detection:
    """One object on one frame, as reported by the segmenter."""

    frame: int
    track_id: int
    box: tuple[float, float, float, float]
    area_px: float
    cx: float
    cy: float
    major_px: float
    minor_px: float
    angle_rad: float
    confidence: float | None = None
    zone_fraction: Mapping[str, float] = field(default_factory=dict)


@dataclass
class Tracklet:
    """All detections the segmenter reported under one object id, in frame order."""

    tracklet_id: int
    detections: list[Detection]

    @property
    def span(self) -> Span:
        return Span(self.detections[0].frame, self.detections[-1].frame + 1)


@dataclass
class SubjectTrack:
    """Per-frame state of one experimental subject over the whole analysed range.

    Arrays have length ``n_frames``. Frames where the subject is absent hold NaN
    in float arrays, ``False`` in ``present`` and ``-1`` in ``source``.
    ``interpolated`` marks frames filled in across a short gap rather than observed.
    ``x_cm``/``y_cm`` are filled once the track is projected onto the arena.
    ``zone_fraction[zone_id]`` is the fraction of the subject's body (mask pixels)
    inside that zone on each frame.
    """

    subject: int
    fps: float
    present: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    area_px: np.ndarray
    major_px: np.ndarray
    minor_px: np.ndarray
    angle_rad: np.ndarray
    confidence: np.ndarray
    source: np.ndarray
    interpolated: np.ndarray
    x_cm: np.ndarray | None = None
    y_cm: np.ndarray | None = None
    zone_fraction: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.present.shape[0])

    @classmethod
    def empty(cls, subject: int, n_frames: int, fps: float) -> SubjectTrack:
        nan = lambda: np.full(n_frames, np.nan)
        return cls(
            subject=subject,
            fps=fps,
            present=np.zeros(n_frames, dtype=bool),
            cx=nan(),
            cy=nan(),
            area_px=nan(),
            major_px=nan(),
            minor_px=nan(),
            angle_rad=nan(),
            confidence=nan(),
            source=np.full(n_frames, -1, dtype=np.int64),
            interpolated=np.zeros(n_frames, dtype=bool),
        )


class EventKind(str, Enum):
    GAP = "gap"
    TELEPORT = "teleport"
    LOW_CONFIDENCE = "low_confidence"
    COUNT_MISMATCH = "count_mismatch"
    CONTACT = "contact"
    ID_CHURN = "id_churn"
    PROPAGATION_DISAGREEMENT = "propagation_disagreement"
    SCORING_INCOMPLETE = "scoring_incomplete"


class Severity(str, Enum):
    MINOR = "minor"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class IntegrityEvent:
    """Something that makes a stretch of a subject's track untrustworthy.

    ``subjects`` is empty for events that concern the scene rather than a subject
    (e.g. an extra object). Blocking events lock every measurement that depends on
    an affected subject over an overlapping span, until resolved.
    """

    id: str
    kind: EventKind
    severity: Severity
    subjects: tuple[int, ...]
    span: Span
    detail: str
    fix_frame: int | None = None
    resolved_by: str | None = None

    @property
    def is_open_blocker(self) -> bool:
        return self.severity is Severity.BLOCKING and self.resolved_by is None


IDENTITY_KINDS = frozenset({EventKind.ID_CHURN, EventKind.CONTACT})
_FOREVER = 1 << 62


def affected_span(event: IntegrityEvent) -> Span:
    """The frames whose numbers an event can corrupt.

    Most events only spoil their own stretch. An unresolved identity doubt is
    different: after it, the track may belong to the other animal, so every later
    frame of that subject is in question until a person settles it.
    """
    if event.kind in IDENTITY_KINDS:
        return Span(event.span.start, _FOREVER)
    return event.span


def open_blockers(
    events: Iterable[IntegrityEvent], subjects: Iterable[int] | None, span: Span
) -> list[IntegrityEvent]:
    """Unresolved blocking events that touch any of ``subjects`` within ``span``.

    ``subjects=None`` means the measurement depends on the whole scene, so every
    blocking event in the span counts. Scene-level events (no subjects) always count.
    Identity events reach from their start to the end of the recording.
    """
    wanted = None if subjects is None else set(subjects)
    hits = []
    for event in events:
        if not event.is_open_blocker or not affected_span(event).overlaps(span):
            continue
        if wanted is None or not event.subjects or wanted.intersection(event.subjects):
            hits.append(event)
    return hits


class Status(str, Enum):
    VERIFIED = "verified"
    LOCKED = "locked"
    REFUSED = "refused"


@dataclass(frozen=True)
class Evidence:
    """A stretch of video that supports (part of) a reported number."""

    span: Span
    subject: int | None
    note: str


@dataclass(frozen=True)
class MetricResult:
    """One reported number.

    ``value`` is only set when ``status`` is VERIFIED; a locked number is withheld
    and its would-be value is kept in ``provisional`` for audit and evaluation only.
    ``reasons`` lists blocking event ids (LOCKED) or a plain-language reason (REFUSED).
    ``uncertainty`` maps a named component to a ± half-width in ``unit``.
    """

    metric_id: str
    subject: int | None
    bin: Span | None
    unit: str
    status: Status
    value: float | None = None
    provisional: float | None = None
    reasons: tuple[str, ...] = ()
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    evidence: tuple[Evidence, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is Status.VERIFIED and self.value is None:
            raise ValueError(f"{self.metric_id}: a verified result needs a value")
        if self.status is not Status.VERIFIED and self.value is not None:
            raise ValueError(f"{self.metric_id}: only verified results may carry a value")
        if self.status is not Status.VERIFIED and not self.reasons:
            raise ValueError(f"{self.metric_id}: a {self.status.value} result needs a reason")


@dataclass(frozen=True)
class BehaviourEvent:
    """One bout of a named behaviour by one subject."""

    behaviour: str
    subject: int
    span: Span
    confidence: float
    source: str
    rationale: str = ""
