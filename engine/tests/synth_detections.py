"""Synthetic segmenter output for testing subject assembly and integrity checks.

Ground truth is a set of per-subject centre paths in image pixels (``(n_frames, 2)``
arrays, NaN rows where the object is out of view). :class:`Synth` turns them into
the detections a segmenter would report and then injects one fault at a time, so a
test can say exactly which fault each integrity event is answering.

Defaults model a mouse filmed top-down: 10 px/cm, 30 fps, an 8 cm x 4 cm body,
walking at 10-20 cm/s.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kappa.model import Detection

FPS = 30.0
PX_PER_CM = 10.0
BODY_PX = 80.0
WIDTH_PX = 40.0


def _speed_px_per_frame(speed_cm_s: float, fps: float, px_per_cm: float) -> float:
    return speed_cm_s * px_per_cm / fps


def line(n: int, start: tuple[float, float], velocity: tuple[float, float]) -> np.ndarray:
    """Straight path from ``start`` moving ``velocity`` px/frame."""
    t = np.arange(n, dtype=float)[:, None]
    return np.asarray(start, dtype=float) + t * np.asarray(velocity, dtype=float)


def still(n: int, at: tuple[float, float]) -> np.ndarray:
    return np.tile(np.asarray(at, dtype=float), (n, 1))


def waypoints(n: int, points: list[tuple[int, float, float]]) -> np.ndarray:
    """Piecewise-straight path through ``(frame, x, y)`` points, held at the ends."""
    frames, xs, ys = (np.asarray(v, dtype=float) for v in zip(*points, strict=True))
    t = np.arange(n, dtype=float)
    return np.column_stack([np.interp(t, frames, xs), np.interp(t, frames, ys)])


def wander(
    n: int,
    seed: int,
    *,
    centre: tuple[float, float] = (400.0, 400.0),
    speed_cm_s: float = 15.0,
    reach_px: float = 150.0,
    fps: float = FPS,
    px_per_cm: float = PX_PER_CM,
) -> np.ndarray:
    """Smooth random path whose speed never exceeds ``speed_cm_s``.

    Each axis is a sum of three sinusoids; amplitudes are scaled so the summed
    amplitude-times-frequency bound keeps the speed under the limit, and the
    excursion is capped at ``reach_px`` so the path stays inside its region.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    vmax = _speed_px_per_frame(speed_cm_s, fps, px_per_cm) / np.sqrt(2.0)
    out = np.empty((n, 2))
    for axis in range(2):
        omega = rng.uniform(2 * np.pi * 0.05, 2 * np.pi * 0.5, size=3) / fps
        phase = rng.uniform(0, 2 * np.pi, size=3)
        weight = rng.uniform(0.2, 1.0, size=3)
        amp = vmax * weight / weight.sum() / omega
        amp *= min(1.0, reach_px / amp.sum())
        out[:, axis] = centre[axis] + (
            amp[:, None] * np.sin(omega[:, None] * t + phase[:, None])
        ).sum(0)
    return out


def crossing(
    n: int,
    *,
    centre: tuple[float, float] = (400.0, 400.0),
    cross_frame: int | None = None,
    lag_frames: int = 4,
    speed_cm_s: float = 15.0,
    fps: float = FPS,
    px_per_cm: float = PX_PER_CM,
) -> tuple[np.ndarray, np.ndarray]:
    """A walks left to right and B top to bottom; they pass through each other.

    A crosses ``centre`` at ``cross_frame``, B ``lag_frames`` later, so the two never
    sit on exactly the same pixel and the moment of closest approach is unambiguous.
    """
    cross = n // 2 if cross_frame is None else cross_frame
    v = _speed_px_per_frame(speed_cm_s, fps, px_per_cm)
    t = np.arange(n, dtype=float)
    cx, cy = centre
    a = np.column_stack([cx + v * (t - cross), np.full(n, cy)])
    b = np.column_stack([np.full(n, cx), cy + v * (t - cross - lag_frames)])
    return a, b


def closest_approach(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.nanargmin(np.hypot(*(a - b).T)))


def touch_and_turn_back(
    n: int,
    *,
    centre: tuple[float, float] = (400.0, 400.0),
    meet_frame: int | None = None,
    pause_frames: int = 10,
    stop_px: float = 25.0,
    speed_cm_s: float = 15.0,
    fps: float = FPS,
    px_per_cm: float = PX_PER_CM,
) -> tuple[np.ndarray, np.ndarray]:
    """A comes in from the left and B from above; they touch, pause, and each retreats.

    Kinematically this is the hard case: after the touch, each animal leaves along the
    line the *other* one would have continued on, so motion continuity alone cannot
    say whether they passed through each other or turned back.
    """
    meet = n // 2 if meet_frame is None else meet_frame
    v = _speed_px_per_frame(speed_cm_s, fps, px_per_cm)
    t = np.arange(n, dtype=float)
    away = np.where(
        t < meet, meet - t, np.where(t < meet + pause_frames, 0.0, t - meet - pause_frames)
    )
    offset = stop_px + v * away
    cx, cy = centre
    a = np.column_stack([cx - offset, np.full(n, cy)])
    b = np.column_stack([np.full(n, cx), cy - offset])
    return a, b


def _axis_angle(path: np.ndarray, frame: int) -> float:
    """Body axis orientation from the direction of travel, folded into (-pi/2, pi/2]."""
    lo, hi = max(frame - 1, 0), min(frame + 1, len(path) - 1)
    d = path[hi] - path[lo]
    if not np.all(np.isfinite(d)) or np.hypot(*d) < 1e-9:
        return 0.0
    angle = float(np.arctan2(d[1], d[0]))
    if angle <= -np.pi / 2:
        angle += np.pi
    elif angle > np.pi / 2:
        angle -= np.pi
    return angle


def detection(
    frame: int,
    track_id: int,
    cx: float,
    cy: float,
    *,
    major_px: float = BODY_PX,
    minor_px: float = WIDTH_PX,
    angle_rad: float = 0.0,
    confidence: float | None = 0.9,
    zone_fraction: dict[str, float] | None = None,
) -> Detection:
    a, b = major_px / 2, minor_px / 2
    hw = float(np.hypot(a * np.cos(angle_rad), b * np.sin(angle_rad)))
    hh = float(np.hypot(a * np.sin(angle_rad), b * np.cos(angle_rad)))
    return Detection(
        frame=frame,
        track_id=track_id,
        box=(cx - hw, cy - hh, cx + hw, cy + hh),
        area_px=float(np.pi / 4 * major_px * minor_px),
        cx=float(cx),
        cy=float(cy),
        major_px=major_px,
        minor_px=minor_px,
        angle_rad=angle_rad,
        confidence=confidence,
        zone_fraction=dict(zone_fraction or {}),
    )


class Synth:
    """Builds a segmenter's detections from ground-truth paths, then breaks them.

    Every fault method returns ``self`` so a scenario reads as one expression.
    ``truth[track_id]`` keeps the clean path each original track id was made from.
    """

    def __init__(
        self,
        n_frames: int,
        *,
        fps: float = FPS,
        px_per_cm: float = PX_PER_CM,
        jitter_px: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.n_frames = n_frames
        self.fps = fps
        self.px_per_cm = px_per_cm
        self.jitter_px = jitter_px
        self.truth: dict[int, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)
        self._dets: list[Detection] = []

    # -- objects -----------------------------------------------------------------

    def add(
        self,
        track_id: int,
        path: np.ndarray,
        *,
        major_px: float = BODY_PX,
        minor_px: float = WIDTH_PX,
        confidence: float | None = 0.9,
        zones: dict[str, float] | None = None,
    ) -> Synth:
        """A subject following ``path`` (NaN rows = out of view), reported as ``track_id``."""
        if path.shape != (self.n_frames, 2):
            raise ValueError(f"path must be ({self.n_frames}, 2), got {path.shape}")
        self.truth[track_id] = path.copy()
        noise = self._rng.normal(0.0, self.jitter_px, size=path.shape) if self.jitter_px else 0.0
        observed = path + noise
        for f in range(self.n_frames):
            if not np.all(np.isfinite(path[f])):
                continue
            self._dets.append(
                detection(
                    f,
                    track_id,
                    float(observed[f, 0]),
                    float(observed[f, 1]),
                    major_px=major_px,
                    minor_px=minor_px,
                    angle_rad=_axis_angle(path, f),
                    confidence=confidence,
                    zone_fraction=zones,
                )
            )
        return self

    def add_static(
        self,
        track_id: int,
        at: tuple[float, float],
        *,
        frames: tuple[int, int] | None = None,
        major_px: float = 30.0,
        minor_px: float = 20.0,
        confidence: float | None = 0.6,
    ) -> Synth:
        """A motionless non-subject (a reflection, a dropping) seen over ``frames``."""
        start, end = frames or (0, self.n_frames)
        path = np.full((self.n_frames, 2), np.nan)
        path[start:end] = at
        return self.add(track_id, path, major_px=major_px, minor_px=minor_px, confidence=confidence)

    # -- faults ------------------------------------------------------------------

    def _edit(self, track_id: int, start: int, end: int | None, fn) -> Synth:
        end = self.n_frames if end is None else end
        out = []
        for d in self._dets:
            if d.track_id == track_id and start <= d.frame < end:
                d = fn(d)
                if d is None:
                    continue
            out.append(d)
        self._dets = out
        return self

    def drop(self, track_id: int, start: int, end: int) -> Synth:
        """The segmenter misses the object on frames ``[start, end)``."""
        return self._edit(track_id, start, end, lambda d: None)

    def reissue(self, track_id: int, from_frame: int, new_id: int) -> Synth:
        """From ``from_frame`` on, the same object is reported under ``new_id``."""
        return self._edit(track_id, from_frame, None, lambda d: replace(d, track_id=new_id))

    def teleport(self, track_id: int, frame: int, offset_px: tuple[float, float]) -> Synth:
        """On one frame the detection jumps to a wrong place (e.g. a wrong object)."""
        dx, dy = offset_px

        def shift(d: Detection) -> Detection:
            x0, y0, x1, y1 = d.box
            return replace(d, cx=d.cx + dx, cy=d.cy + dy, box=(x0 + dx, y0 + dy, x1 + dx, y1 + dy))

        return self._edit(track_id, frame, frame + 1, shift)

    def low_confidence(self, track_id: int, start: int, end: int, value: float = 0.2) -> Synth:
        return self._edit(track_id, start, end, lambda d: replace(d, confidence=value))

    def strip_confidence(self, track_id: int) -> Synth:
        """The segmenter reports no confidence at all for this object."""
        return self._edit(track_id, 0, None, lambda d: replace(d, confidence=None))

    def swap_ids(self, id_a: int, id_b: int, from_frame: int) -> Synth:
        """The segmenter exchanges two objects' ids from ``from_frame`` on."""
        swap = {id_a: id_b, id_b: id_a}
        self._dets = [
            replace(d, track_id=swap[d.track_id])
            if d.track_id in swap and d.frame >= from_frame
            else d
            for d in self._dets
        ]
        return self

    def duplicate(
        self,
        track_id: int,
        frame: int,
        offset_px: tuple[float, float],
        confidence: float | None = 0.95,
    ) -> Synth:
        """A second detection under the same id on one frame (e.g. a split mask)."""
        src = next(d for d in self._dets if d.track_id == track_id and d.frame == frame)
        self._dets.append(
            replace(src, cx=src.cx + offset_px[0], cy=src.cy + offset_px[1], confidence=confidence)
        )
        return self

    # -- output ------------------------------------------------------------------

    def detections(self) -> list[Detection]:
        return sorted(self._dets, key=lambda d: (d.frame, d.track_id))

    def centres(self, track_id: int) -> np.ndarray:
        """The (noisy) centres actually reported under ``track_id``, NaN where absent."""
        out = np.full((self.n_frames, 2), np.nan)
        for d in self._dets:
            if d.track_id == track_id:
                out[d.frame] = (d.cx, d.cy)
        return out
