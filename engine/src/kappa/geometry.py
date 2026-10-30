"""Image pixels to arena centimetres, and where the zones lie.

The camera is fixed, so one homography from the located floor corners maps every
frame onto the arena. Zones are defined in arena centimetres, which keeps them
independent of where the camera happened to sit; ``inside`` answers exactly for a
point, and ``zone_raster`` paints the same zone back into the image so it can be
intersected with segmentation masks.
"""

from __future__ import annotations

import copy
import math

import cv2
import numpy as np
from numpy.typing import ArrayLike

from kappa.model import SubjectTrack
from kappa.protocol import ArenaSpec, ZoneSpec

CIRCLE_VERTICES = 256
_SUBPIXEL_BITS = 8


def arena_anchors_cm(arena: ArenaSpec) -> np.ndarray:
    """Where the four ``corners_px`` points sit on the arena floor, in the same order."""
    w, h = arena.width_cm, arena.height_cm
    if arena.shape == "rect":
        return np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=float)
    r = w / 2
    return np.array([(0, r), (r, 0), (2 * r, r), (r, 2 * r)], dtype=float)


class Calibration:
    """The homography between one camera view and its arena floor.

    ``px_per_cm`` is a single representative scale (the mean over the four anchor
    edges of pixel length per centimetre). Under perspective the true scale varies
    across the floor, so it suits tolerances and sizes, never measurements.
    """

    def __init__(self, arena: ArenaSpec) -> None:
        if arena.corners_px is None:
            raise ValueError("arena.corners_px is not set; locate the arena on a keyframe first")
        src = np.asarray(arena.corners_px, dtype=float)
        dst = arena_anchors_cm(arena)
        if not _convex_in_order(src):
            raise ValueError(
                "arena.corners_px must form a convex quadrilateral listed in order "
                "(top-left, top-right, bottom-right, bottom-left; or left, top, right, bottom rim)"
            )
        homography = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
            raise ValueError("arena.corners_px are degenerate")
        self.arena = arena
        self.px_to_cm = homography
        self.cm_to_px = np.linalg.inv(homography)
        edges_px = np.linalg.norm(np.roll(src, -1, axis=0) - src, axis=1)
        edges_cm = np.linalg.norm(np.roll(dst, -1, axis=0) - dst, axis=1)
        self.px_per_cm = float(np.mean(edges_px / edges_cm))

    def to_cm(self, x_px: ArrayLike, y_px: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        return _apply(self.px_to_cm, x_px, y_px)

    def to_px(self, x_cm: ArrayLike, y_cm: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        return _apply(self.cm_to_px, x_cm, y_cm)


def _apply(h: np.ndarray, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w


def _convex_in_order(quad: np.ndarray) -> bool:
    """A wrong corner order shows up as a self-intersecting (non-convex) quadrilateral."""
    edges = np.roll(quad, -1, axis=0) - quad
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    return bool((turns > 0).all() or (turns < 0).all())


def project(track: SubjectTrack, calib: Calibration) -> SubjectTrack:
    """A copy of ``track`` with arena coordinates filled in from its pixel centroids."""
    out = copy.deepcopy(track)
    x, y = calib.to_cm(track.cx, track.cy)
    out.x_cm = np.where(track.present, x, np.nan)
    out.y_cm = np.where(track.present, y, np.nan)
    return out


def _half_extents(zone: ZoneSpec, arena: ArenaSpec) -> tuple[float, float] | None:
    """Half-width and half-height of the centred shape that bounds a zone from inside.

    For ``center_fraction`` that is the zone itself; for ``periphery_band`` it is the
    inner region the band surrounds (``None`` once the band swallows the whole floor).
    """
    hw, hh = arena.width_cm / 2, arena.height_cm / 2
    if zone.kind == "center_fraction":
        s = math.sqrt(zone.fraction)
        return hw * s, hh * s
    if zone.kind == "periphery_band":
        a, b = hw - zone.band_cm, hh - zone.band_cm
        return (a, b) if a > 0 and b > 0 else None
    raise ValueError(f"zone {zone.id}: {zone.kind} is not a centred shape")


def _in_centred(arena: ArenaSpec, a: float, b: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx, dy = x - arena.width_cm / 2, y - arena.height_cm / 2
    if arena.shape == "rect":
        return (np.abs(dx) <= a) & (np.abs(dy) <= b)
    return (dx / a) ** 2 + (dy / b) ** 2 <= 1.0


def _sorted_rect(zone: ZoneSpec) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = zone.rect_cm
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def inside(zone: ZoneSpec, arena: ArenaSpec, x_cm: ArrayLike, y_cm: ArrayLike) -> np.ndarray:
    """Which points lie in ``zone``. Exact; points off the arena floor are in no zone.

    Boundaries count as inside the arena and the centre; the band is the floor minus
    its (closed) inner region, so centre and band partition the floor when they meet.
    """
    x, y = np.broadcast_arrays(np.asarray(x_cm, dtype=float), np.asarray(y_cm, dtype=float))
    floor = _in_centred(arena, arena.width_cm / 2, arena.height_cm / 2, x, y)
    if zone.kind == "center_fraction":
        return floor & _in_centred(arena, *_half_extents(zone, arena), x, y)
    if zone.kind == "periphery_band":
        inner = _half_extents(zone, arena)
        return floor if inner is None else floor & ~_in_centred(arena, *inner, x, y)
    if zone.kind == "rect_cm":
        x0, y0, x1, y1 = _sorted_rect(zone)
        return floor & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    raise ValueError(f"zone {zone.id}: unsupported kind {zone.kind}")


def _centred_polygon(arena: ArenaSpec, a: float, b: float) -> np.ndarray:
    cx, cy = arena.width_cm / 2, arena.height_cm / 2
    if arena.shape == "rect":
        return np.array([(cx - a, cy - b), (cx + a, cy - b), (cx + a, cy + b), (cx - a, cy + b)])
    t = np.linspace(0.0, 2 * np.pi, CIRCLE_VERTICES, endpoint=False)
    return np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])


def _fill(polygon_cm: np.ndarray, calib: Calibration, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    px, py = calib.to_px(polygon_cm[:, 0], polygon_cm[:, 1])
    points = np.round(np.column_stack([px, py]) * (1 << _SUBPIXEL_BITS)).astype(np.int32)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8, shift=_SUBPIXEL_BITS)
    return mask


def zone_raster(
    zone: ZoneSpec, arena: ArenaSpec, calib: Calibration, width: int, height: int
) -> np.ndarray:
    """The zone painted into a ``height × width`` image: 1 inside, 0 outside.

    Straight arena edges stay straight under a homography, so rectangles map exactly;
    circles are drawn as ``CIRCLE_VERTICES``-gons, well under a pixel from the true rim.
    """
    floor = _fill(
        _centred_polygon(arena, arena.width_cm / 2, arena.height_cm / 2), calib, width, height
    )
    if zone.kind == "center_fraction":
        region = _fill(_centred_polygon(arena, *_half_extents(zone, arena)), calib, width, height)
    elif zone.kind == "periphery_band":
        inner = _half_extents(zone, arena)
        region = (
            floor
            if inner is None
            else floor & (1 - _fill(_centred_polygon(arena, *inner), calib, width, height))
        )
    elif zone.kind == "rect_cm":
        x0, y0, x1, y1 = _sorted_rect(zone)
        region = _fill(
            np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=float), calib, width, height
        )
    else:
        raise ValueError(f"zone {zone.id}: unsupported kind {zone.kind}")
    return floor & region
