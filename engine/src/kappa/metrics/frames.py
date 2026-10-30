"""Frame bookkeeping shared by every metric: the analysed window, runs and bins."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from kappa.model import Span, SubjectTrack


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of ``True`` as half-open ``(start, end)`` index pairs."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def frames_for(seconds: float, fps: float) -> int:
    """The fewest whole frames that last at least ``seconds``.

    The small slack keeps ``1.0 s`` at 25 fps at 25 frames despite float rounding.
    """
    return max(0, math.ceil(seconds * fps - 1e-9))


def bins(span: Span, bin_s: float, fps: float) -> list[Span]:
    """Consecutive bins of ``round(bin_s * fps)`` frames over ``span``; the last may be short."""
    size = max(1, round(bin_s * fps))
    return [Span(s, min(s + size, span.end)) for s in range(span.start, span.end, size)]


def _take(values: np.ndarray, span: Span, fill) -> np.ndarray:
    out = np.full(span.length, fill, dtype=values.dtype)
    stop = min(span.end, values.shape[0])
    if stop > span.start:
        out[: stop - span.start] = values[span.start : stop]
    return out


@dataclass(frozen=True, eq=False)
class SubjectView:
    """One subject's per-frame arrays over the analysed span only.

    Everything a metric reports is computed from inside this window, so a number never
    depends on frames its integrity gate did not look at. Frames past the end of the
    recording count as absent, so an over-long span measures what was seen instead of
    failing.
    """

    subject: int
    fps: float
    span: Span
    present: np.ndarray
    x: np.ndarray | None
    y: np.ndarray | None
    interpolated: np.ndarray
    zone_fraction: Mapping[str, np.ndarray]

    @classmethod
    def of(cls, track: SubjectTrack, span: Span) -> SubjectView:
        present = _take(track.present.astype(bool), span, False)
        coords = None
        if track.x_cm is not None and track.y_cm is not None:
            coords = tuple(
                np.where(present, _take(np.asarray(v, dtype=float), span, np.nan), np.nan)
                for v in (track.x_cm, track.y_cm)
            )
        return cls(
            subject=track.subject,
            fps=float(track.fps),
            span=span,
            present=present,
            x=None if coords is None else coords[0],
            y=None if coords is None else coords[1],
            interpolated=_take(track.interpolated.astype(bool), span, False) & present,
            zone_fraction={
                zone: np.where(present, _take(np.asarray(v, dtype=float), span, np.nan), np.nan)
                for zone, v in track.zone_fraction.items()
            },
        )

    def local(self, window: Span) -> slice:
        return slice(window.start - self.span.start, window.end - self.span.start)

    def spans(self, mask: np.ndarray, window: Span | None = None) -> list[Span]:
        """The runs of a per-frame mask over this view, as absolute frame spans.

        With ``window``, only the part of the mask inside it, so runs are cut at its edges.
        """
        window = window or self.span
        offset = window.start
        return [Span(offset + s, offset + e) for s, e in runs(mask[self.local(window)])]

    def observed_in(self, window: Span) -> bool:
        return bool(self.present[self.local(window)].any())
