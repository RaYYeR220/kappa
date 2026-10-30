"""Turn SAM 3.1's raw output text into per-frame detections.

The raw text (the concatenated ``output_text`` lane) is the canonical artifact of a
tracking run: it is stored content-addressed and re-parsed offline, so a result can
always be traced back to exactly what the model returned.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationDiagnostic,
    SegmentationMask,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    video_segmentation_format,
)

from kappa.features import mask_features
from kappa.model import Detection


@dataclass(frozen=True)
class ParsedOutput:
    detections: list[Detection]
    width: int | None
    height: int | None
    diagnostics: tuple[SegmentationDiagnostic, ...]


@dataclass(frozen=True)
class _Pending:
    frame: int
    track_id: int
    box: tuple[int, int, int, int]
    confidence: float | None
    encoding: str
    payload: str
    mask_w: int
    mask_h: int


def _decode(item: tuple[str, str, int, int]) -> np.ndarray:
    encoding, payload, width, height = item
    raster = decode_mask_to_raster(
        SegmentationMask(encoding=encoding, payload=payload, width=width, height=height)
    )
    return np.frombuffer(raster, dtype=np.uint8).reshape(height, width)


def _records(raw: str, media: str):
    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(raw, emit=False)
    return parser.finish(CompletedOutcome()).result


_SIZE = re.compile(r";w=(\d+);h=(\d+)")


def _box_size(raw: str) -> tuple[int | None, int | None]:
    match = _SIZE.search(raw)
    return (int(match[1]), int(match[2])) if match else (None, None)


def parse_output(
    raw: str,
    media: str = "video",
    zone_rasters: Mapping[str, np.ndarray] | None = None,
    workers: int | None = None,
) -> ParsedOutput:
    """Parse raw SAM output into detections sorted by (frame, track id).

    Boxes are ``(left, top, right, bottom)`` with right and bottom exclusive, as the
    reference parser reports them (the wire format is inclusive).

    When a mask for the same frame and object arrives more than once (later
    revisions from prompt propagation), the last one wins. Mask decoding is pure
    Python and dominates the cost on long clips, so it fans out across processes.
    """
    result = _records(raw, media)
    boxes: dict[tuple[int, int], SegmentationBoxRecord] = {}
    masks: dict[tuple[int, int], SegmentationMaskRecord] = {}
    for record in result.records:
        if isinstance(record, (SegmentationBoxRecord, SegmentationMaskRecord)):
            frame = record.frame.frame_index if record.frame is not None else 0
            key = (frame, int(record.object_id))
            (boxes if isinstance(record, SegmentationBoxRecord) else masks)[key] = record

    pending: list[_Pending] = []
    for key in sorted(boxes.keys() & masks.keys()):
        box, mask = boxes[key], masks[key]
        pending.append(
            _Pending(
                frame=key[0],
                track_id=key[1],
                box=(int(box.left), int(box.top), int(box.right), int(box.bottom)),
                confidence=box.confidence if box.confidence is not None else mask.confidence,
                encoding=mask.mask.encoding,
                payload=mask.mask.payload,
                mask_w=int(mask.mask.width),
                mask_h=int(mask.mask.height),
            )
        )

    items = [(p.encoding, p.payload, p.mask_w, p.mask_h) for p in pending]
    if workers is None:
        workers = min(8, os.cpu_count() or 1) if len(items) > 200 else 1
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rasters = list(pool.map(_decode, items, chunksize=64))
    else:
        rasters = [_decode(item) for item in items]

    detections = []
    for p, raster in zip(pending, rasters, strict=True):
        feats = mask_features(raster, p.box[0], p.box[1], zone_rasters)
        if feats is None:
            continue
        detections.append(
            Detection(
                frame=p.frame,
                track_id=p.track_id,
                box=tuple(float(v) for v in p.box),
                area_px=feats.area_px,
                cx=feats.cx,
                cy=feats.cy,
                major_px=feats.major_px,
                minor_px=feats.minor_px,
                angle_rad=feats.angle_rad,
                confidence=p.confidence,
                zone_fraction=dict(feats.zone_fraction),
            )
        )
    width, height = _box_size(raw)
    return ParsedOutput(detections, width, height, tuple(result.diagnostics))
