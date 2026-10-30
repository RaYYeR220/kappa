"""Synthetic, already-projected subject tracks with known answers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from kappa.model import EventKind, IntegrityEvent, Severity, Span, SubjectTrack
from kappa.protocol import AssayProtocol

FPS = 25.0
CENTRE = (22.5, 22.5)
OUTSIDE = (5.0, 5.0)


def track(
    x: Sequence[float],
    y: Sequence[float],
    *,
    subject: int = 1,
    fps: float = FPS,
    interpolated: Sequence[int] = (),
    zone_fraction: Mapping[str, Sequence[float]] | None = None,
) -> SubjectTrack:
    """A track whose arena coordinates are ``x``/``y`` (NaN = absent); pixels mirror cm."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    t = SubjectTrack.empty(subject, len(x), fps)
    present = np.isfinite(x) & np.isfinite(y)
    t.present[:] = present
    t.cx[:], t.cy[:] = np.where(present, x, np.nan), np.where(present, y, np.nan)
    t.area_px[present] = 30.0
    t.major_px[present], t.minor_px[present], t.angle_rad[present] = 8.0, 4.0, 0.0
    t.confidence[present] = 0.95
    t.source[present] = subject
    t.interpolated[list(interpolated)] = True
    t.x_cm, t.y_cm = t.cx.copy(), t.cy.copy()
    for zone, fraction in (zone_fraction or {}).items():
        t.zone_fraction[zone] = np.where(present, np.asarray(fraction, dtype=float), np.nan)
    return t


def line(n: int, start=(5.0, 22.5), velocity=(10.0, 0.0), fps: float = FPS):
    """Straight line at constant velocity (cm/s)."""
    t = np.arange(n) / fps
    return start[0] + velocity[0] * t, start[1] + velocity[1] * t


def circle_path(n: int, centre=CENTRE, radius=15.0, period_s=8.0, fps: float = FPS):
    t = np.arange(n) / fps
    a = 2 * np.pi * t / period_s
    return centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)


def jitter(n: int, at=CENTRE, amplitude=0.2, seed=0):
    """A subject sitting still, with tracker noise of up to ``amplitude`` cm per axis."""
    noise = np.random.default_rng(seed).uniform(-amplitude, amplitude, (2, n))
    return at[0] + noise[0], at[1] + noise[1]


def dwell(*legs: tuple[tuple[float, float], int]):
    """Jump between fixed points: ``((x, y), frames)`` legs, e.g. outside → centre → outside."""
    x = np.concatenate([np.full(k, p[0]) for p, k in legs])
    y = np.concatenate([np.full(k, p[1]) for p, k in legs])
    return x, y


def through_centre(n: int, fps: float = FPS):
    """Wall to wall along the midline, so the centre zone is crossed once."""
    return line(n, start=(1.0, 22.5), velocity=(43.0 / ((n - 1) / fps), 0.0), fps=fps)


CENTRE_ZONE = {"id": "center", "kind": "center_fraction", "fraction": 0.25}
REARING = {
    "id": "rearing",
    "label": "Rearing",
    "definition": "Standing on the hind legs with the forepaws off the floor.",
    "detector": "rearing_topdown",
}


def protocol(
    metrics, *, zones=None, entry_rule=None, smoothing=None, subjects=1, arena=None
) -> AssayProtocol:
    """An open-field protocol; smoothing defaults to none so expected values stay exact."""
    return AssayProtocol.model_validate(
        {
            "title": "Open field",
            "question": "How does the mouse use the open field?",
            "subject": {"concept": "mouse", "expected_count": subjects},
            "arena": arena or {"shape": "rect", "width_cm": 45, "height_cm": 45},
            "zones": [CENTRE_ZONE] if zones is None else zones,
            "entry_rule": entry_rule or {},
            "metrics": metrics,
            "smoothing": smoothing or {"method": "none"},
            "behaviours": [REARING],
        }
    )


def blocker(
    span: Span,
    subjects: tuple[int, ...] = (1,),
    *,
    severity: Severity = Severity.BLOCKING,
    resolved_by: str | None = None,
    kind: EventKind = EventKind.TELEPORT,
) -> IntegrityEvent:
    return IntegrityEvent(
        id=f"{kind.value}:{','.join(map(str, subjects))}:{span.start}",
        kind=kind,
        severity=severity,
        subjects=subjects,
        span=span,
        detail="synthetic",
        resolved_by=resolved_by,
    )
