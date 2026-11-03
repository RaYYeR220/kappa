"""Per-frame silhouette descriptors of one subject, from its SAM 3.1 masks.

A top-down camera sees a rear as a change of silhouette: the body foreshortens, the
outline rounds and the convex hull tightens. Area and ellipse axes alone carry little
of that, so these descriptors add perimeter, solidity, extent and Hu moments, each
then normalised by the animal's own rolling typical value so one model fits animals
and cameras of different sizes.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from kappa.sam.fastmask import decode_mask
from kappa.sam.parse import _pending_fast

SILHOUETTE = ("area", "perimeter", "solidity", "extent", "major", "minor", "radius", "hu0", "hu1", "hu2")


def _describe(mask: np.ndarray) -> list[float]:
    m = mask.astype(np.uint8)
    area = float(m.sum())
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours or area < 3:
        return [np.nan] * len(SILHOUETTE)
    c = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(c, True))
    hull = float(cv2.contourArea(cv2.convexHull(c))) or area
    h, w = m.shape
    moments = cv2.moments(m, binaryImage=True)
    hu = cv2.HuMoments(moments).ravel()
    cov = np.array([[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]]) / max(moments["m00"], 1)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    radius = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).max()))
    return [
        area,
        perimeter,
        area / hull,
        area / (h * w),
        4 * np.sqrt(max(ev[0], 0)),
        4 * np.sqrt(max(ev[1], 0)),
        radius,
        *(-np.sign(hu[:3]) * np.log10(np.abs(hu[:3]) + 1e-12)),
    ]


def silhouettes(raw: str, n_frames: int, track_id: int | None = None, source: np.ndarray | None = None) -> np.ndarray:
    """``(n_frames, len(SILHOUETTE))`` descriptors of one subject (NaN where absent).

    ``source`` gives the SAM object id behind the subject on each frame (a subject
    track's ``source``), which follows the subject through re-identifications. Without
    it, ``track_id`` names one SAM object, and with neither the object seen on most
    frames is used, which for a single-animal assay is the animal.
    """
    found, _ = _pending_fast(raw)
    if source is not None:
        out = np.full((n_frames, len(SILHOUETTE)), np.nan)
        for frame in range(min(n_frames, len(source))):
            p = found.get((frame, int(source[frame])))
            if source[frame] >= 0 and p is not None:
                out[frame] = _describe(decode_mask(p.encoding, p.payload, p.mask_w, p.mask_h))
        return out
    if track_id is None:
        counts: dict[int, int] = {}
        for _, tid in found:
            counts[tid] = counts.get(tid, 0) + 1
        track_id = max(counts, key=counts.get) if counts else -1
    out = np.full((n_frames, len(SILHOUETTE)), np.nan)
    for (frame, tid), p in found.items():
        if tid == track_id and 0 <= frame < n_frames:
            out[frame] = _describe(decode_mask(p.encoding, p.payload, p.mask_w, p.mask_h))
    return out


def _rel(x: np.ndarray, window: int) -> np.ndarray:
    filled = np.where(np.isfinite(x), x, np.nanmedian(x))
    return x / median_filter(filled, size=window, mode="nearest")


def frame_features(
    shape: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    arena_cm: tuple[float, float],
    fps: float,
    context_s: Sequence[float] = (0.2, 0.6, 1.2),
) -> np.ndarray:
    """Per-frame feature matrix: normalised silhouette, motion, wall distance, and their context."""
    base = max(3, int(20 * fps) | 1)
    rel = np.column_stack([_rel(shape[:, i], base) for i in range(7)] + [shape[:, 7:10]])
    x = np.where(np.isfinite(x_cm), x_cm, np.nan)
    y = np.where(np.isfinite(y_cm), y_cm, np.nan)
    speed = np.hypot(np.gradient(np.nan_to_num(x)), np.gradient(np.nan_to_num(y))) * fps
    w, h = arena_cm
    wall = np.minimum.reduce([x, w - x, y, h - y])
    cols = [rel, speed[:, None], wall[:, None]]
    filled = np.where(np.isfinite(rel), rel, 1.0)
    for s in context_s:
        k = max(1, round(s * fps))
        cols.append(uniform_filter1d(filled, size=2 * k + 1, axis=0, mode="nearest"))
        cols.append(uniform_filter1d(speed, size=2 * k + 1, mode="nearest")[:, None])
        lag = np.roll(filled, k, axis=0) - np.roll(filled, -k, axis=0)
        cols.append(lag)
    return np.column_stack(cols)
