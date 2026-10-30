"""Shape features of one segmentation mask.

The segmenter returns each mask cropped to its box, so pixel ``(row, col)`` of the
mask is image pixel ``(y1 + row, x1 + col)``. Everything here works in full-image
pixel coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskFeatures:
    area_px: float
    cx: float
    cy: float
    major_px: float
    minor_px: float
    angle_rad: float
    zone_fraction: Mapping[str, float]


def mask_features(
    mask: np.ndarray,
    x1: int,
    y1: int,
    zone_rasters: Mapping[str, np.ndarray] | None = None,
) -> MaskFeatures | None:
    """Area, centroid, ellipse axes and per-zone body fraction of a box-cropped mask.

    Axis lengths are full lengths of the ellipse with the same second moments
    (4·sqrt(eigenvalue) for a uniform ellipse). ``angle_rad`` is the major axis
    direction in image coordinates (y down), in ``(-pi/2, pi/2]``. Returns ``None``
    for an empty mask.
    """
    rows, cols = np.nonzero(mask)
    area = rows.size
    if area == 0:
        return None
    ys = rows.astype(np.float64) + y1
    xs = cols.astype(np.float64) + x1
    cx, cy = xs.mean(), ys.mean()
    if area > 1:
        cov = np.cov(np.vstack([xs - cx, ys - cy]), bias=True)
        evals, evecs = np.linalg.eigh(cov)
        minor_var, major_var = max(evals[0], 0.0), max(evals[1], 0.0)
        vx, vy = evecs[:, 1]
        angle = float(np.arctan2(vy, vx))
        if angle <= -np.pi / 2:
            angle += np.pi
        elif angle > np.pi / 2:
            angle -= np.pi
    else:
        minor_var = major_var = 0.0
        angle = 0.0
    fractions: dict[str, float] = {}
    for zone_id, raster in (zone_rasters or {}).items():
        h, w = raster.shape
        inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        hits = raster[ys[inside].astype(np.intp), xs[inside].astype(np.intp)]
        fractions[zone_id] = float(np.count_nonzero(hits)) / area
    return MaskFeatures(
        area_px=float(area),
        cx=float(cx),
        cy=float(cy),
        major_px=float(4.0 * np.sqrt(major_var)),
        minor_px=float(4.0 * np.sqrt(minor_var)),
        angle_rad=angle,
        zone_fraction=fractions,
    )
