"""Path length, speed, immobility and spacing, from smoothed arena positions.

The path is broken wherever the subject was not seen: smoothing runs on each visible
stretch separately and no step is ever taken across a gap, because a straight line
through an unseen stretch is a guess, not a measurement.

Distance is credited to the frame a step arrives at, and only for steps whose start
lies inside the analysed span. Any window then sums the credits of its own frames,
which is what makes per-bin distances add up exactly to the whole.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from kappa.metrics.frames import SubjectView, frames_for, runs
from kappa.metrics.gating import Measurement, Outcome, Unmeasurable, capped
from kappa.model import Evidence, Span
from kappa.protocol import MetricSpec, Smoothing

SAVGOL_POLYORDER = 2
SPEED_ESTIMATOR = "central difference of the smoothed position"


def window_frames(smoothing: Smoothing, fps: float) -> int:
    """The smoothing window in frames: odd so it is centred, and at least 3."""
    w = max(3, round(smoothing.window_s * fps))
    return w if w % 2 else w + 1


def smooth(
    values: np.ndarray, segments: Sequence[tuple[int, int]], smoothing: Smoothing, fps: float
) -> np.ndarray:
    """Smooth each visible segment on its own.

    A savgol window is shrunk to fit a short segment, and a segment under three frames
    is left as observed, rather than fitting a curve to too few points.
    """
    out = values.copy()
    if smoothing.method == "none":
        return out
    w = window_frames(smoothing, fps)
    for s, e in segments:
        if smoothing.method == "median":
            out[s:e] = median_filter(values[s:e], size=w, mode="nearest")
        elif smoothing.method == "savgol":
            k = min(w, e - s if (e - s) % 2 else e - s - 1)
            if k >= 3:
                out[s:e] = savgol_filter(
                    values[s:e], k, min(SAVGOL_POLYORDER, k - 1), mode="interp"
                )
        else:
            raise ValueError(f"unsupported smoothing method {smoothing.method}")
    return out


@dataclass(frozen=True, eq=False)
class Path:
    """A subject's smoothed path over the analysed span, broken wherever it was unseen.

    ``credit[i]`` is the distance credited on arriving at frame ``i``; ``estimated[i]``
    is the part of it measured from or to an interpolated frame. ``speed`` is NaN on
    frames with no visible neighbour to difference against.
    """

    view: SubjectView
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray
    credit: np.ndarray
    estimated: np.ndarray
    speed: np.ndarray


def trace(view: SubjectView, smoothing: Smoothing) -> Path:
    """Smooth and measure a projected view.

    With ``min_step_cm`` the anchor method suppresses jitter: a step is credited only
    once the position has moved at least ``min_step_cm`` from the last credited point,
    which then becomes the new anchor.
    """
    if view.x is None or view.y is None:
        raise ValueError(f"subject {view.subject} has no arena coordinates to trace")
    valid = view.present & np.isfinite(view.x) & np.isfinite(view.y)
    segments = runs(valid)
    x = smooth(view.x, segments, smoothing, view.fps)
    y = smooth(view.y, segments, smoothing, view.fps)
    credit = np.zeros(valid.shape[0])
    estimated = np.zeros(valid.shape[0])
    speed = np.full(valid.shape[0], np.nan)
    xs, ys, interp = x.tolist(), y.tolist(), view.interpolated.tolist()
    for s, e in segments:
        anchor = s
        for i in range(s + 1, e):
            d = math.hypot(xs[i] - xs[anchor], ys[i] - ys[anchor])
            if d >= smoothing.min_step_cm:
                credit[i] = d
                if interp[i] or interp[anchor]:
                    estimated[i] = d
                anchor = i
        if e - s >= 2:
            speed[s:e] = np.hypot(np.gradient(x[s:e]), np.gradient(y[s:e])) * view.fps
    return Path(view=view, x=x, y=y, valid=valid, credit=credit, estimated=estimated, speed=speed)


def _path_evidence(path: Path, window: Span) -> tuple[Evidence, ...]:
    fps, subject = path.view.fps, path.view.subject
    items = [Evidence(window, subject, f"path measured over {window.length / fps:.1f} s")]
    items += [
        Evidence(gap, subject, f"not visible for {gap.length / fps:.1f} s; excluded from the path")
        for gap in path.view.spans(~path.valid, window)
    ]
    return capped(items)


def distance(path: Path, window: Span) -> Measurement:
    sl = path.view.local(window)
    return Measurement(
        value=float(path.credit[sl].sum()),
        evidence=_path_evidence(path, window),
        uncertainty={"interpolated": float(path.estimated[sl].sum())},
    )


def mean_speed(path: Path, window: Span) -> Measurement:
    """Distance over the whole window's duration, unseen stretches included."""
    travelled = distance(path, window)
    seconds = window.length / path.view.fps
    return Measurement(
        value=travelled.value / seconds,
        evidence=travelled.evidence,
        uncertainty={"interpolated": travelled.uncertainty["interpolated"] / seconds},
    )


def immobile_bouts(path: Path, spec: MetricSpec) -> list[Span]:
    """Stretches where speed stays under the threshold for at least the minimum time."""
    still = path.speed < spec.immobile_below_cm_s
    need = max(1, frames_for(spec.immobile_min_s, path.view.fps))
    return [b for b in path.view.spans(still) if b.length >= need]


def immobility_time(path: Path, spec: MetricSpec, window: Span) -> Measurement:
    fps, subject = path.view.fps, path.view.subject
    parts = [c for b in immobile_bouts(path, spec) if (c := b.intersect(window))]
    return Measurement(
        value=sum(c.length for c in parts) / fps,
        evidence=capped(
            [Evidence(c, subject, f"immobile for {c.length / fps:.1f} s") for c in parts]
        ),
    )


def immobility_bout_count(path: Path, spec: MetricSpec, window: Span) -> Measurement:
    """Bouts are counted in the window where they begin, so binned counts add up."""
    fps, subject = path.view.fps, path.view.subject
    starting = [b for b in immobile_bouts(path, spec) if window.contains(b.start)]
    return Measurement(
        value=float(len(starting)),
        evidence=capped(
            [Evidence(b, subject, f"immobile bout of {b.length / fps:.1f} s") for b in starting]
        ),
    )


def inter_subject_distance(paths: Sequence[Path], window: Span) -> Outcome:
    """Mean over frames, where every subject is visible, of the mean pairwise distance."""
    if len(paths) < 2:
        return Unmeasurable("spacing needs at least two tracked subjects")
    view = paths[0].view
    sl = view.local(window)
    together = np.logical_and.reduce([p.valid[sl] for p in paths])
    if not together.any():
        return Unmeasurable(
            f"the subjects were never observed together in frames [{window.start}, {window.end})"
        )
    pairs = [
        np.hypot(a.x[sl] - b.x[sl], a.y[sl] - b.y[sl])[together]
        for i, a in enumerate(paths)
        for b in paths[i + 1 :]
    ]
    items = [
        Evidence(
            window,
            None,
            f"spacing averaged over {together.sum() / view.fps:.1f} s with every subject visible",
        )
    ]
    every = np.zeros(view.present.shape[0], dtype=bool)
    every[sl] = together
    items += [
        Evidence(
            gap, None, f"not every subject visible for {gap.length / view.fps:.1f} s; excluded"
        )
        for gap in view.spans(~every, window)
    ]
    return Measurement(value=float(np.mean(np.mean(pairs, axis=0))), evidence=capped(items))
