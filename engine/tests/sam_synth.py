"""Build SAM 3.1 output text for tests, using the reference codec's own encoder."""

from __future__ import annotations

import numpy as np
from meta_sam_parser._mask_codec import _encode_raster


def ellipse_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    return (((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2 <= 1).astype(np.uint8)


def record(obj: int, x1: int, y1: int, mask: np.ndarray, frame_w: int, frame_h: int, conf: float | None = None) -> str:
    h, w = mask.shape
    c = f";c={conf}" if conf is not None else ""
    payload = _encode_raster(mask.reshape(-1).tolist(), w, h)
    return (
        f"{obj}<|box;x1={x1};y1={y1};x2={x1 + w - 1};y2={y1 + h - 1};w={frame_w};h={frame_h}{c}|>"
        f"<|mask;x=0;y=0{c};data={h},{w},{payload}|>"
    )


def line(frame: int, records: list[str]) -> str:
    return f"<{frame}f>" + ",".join(records)
