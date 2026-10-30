"""Turn SAM 3.1's raw output text into per-frame detections.

The raw text (the concatenated ``output_text`` lane) is the canonical artifact of a
tracking run: it is stored content-addressed and re-parsed offline, so a result can
always be traced back to exactly what the model returned.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationDiagnostic,
    SegmentationMaskRecord,
    image_segmentation_format,
    video_segmentation_format,
)

from kappa.features import mask_features
from kappa.model import Detection
from kappa.sam.fastmask import decode_mask


@dataclass(frozen=True)
class ParsedOutput:
    detections: list[Detection]
    width: int | None
    height: int | None
    diagnostics: tuple[SegmentationDiagnostic | str, ...]


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
    return decode_mask(encoding, payload, width, height)


def _records(raw: str, media: str):
    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(raw, emit=False)
    return parser.finish(CompletedOutcome()).result


_SIZE = re.compile(r";w=(\d+);h=(\d+)")
_FRAME = re.compile(r"<(\d+)f>")
_RECORD = re.compile(
    r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([0-9.]+))?\|>"
    r"<\|mask;x=0;y=0(?:;c=([0-9.]+))?;data=(\d+),(\d+),([~!][^|]*)\|>"
)
_ENCODING = {"!": "one_bit", "~": "lossless"}


def _pending_fast(raw: str) -> tuple[dict[tuple[int, int], _Pending], list[str]]:
    """Records by the documented line grammar, without decoding any mask.

    The reference parser fully decodes (and re-encodes) every mask while it reads the
    text, which costs seconds per thousand masks; here masks are decoded once, later.
    A line that the grammar doesn't fully account for is reported, not guessed at.
    """
    found: dict[tuple[int, int], _Pending] = {}
    problems: list[str] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        head = _FRAME.match(line)
        if head is None:
            problems.append(f"line {number}: no frame marker")
            continue
        frame, pos = int(head[1]), head.end()
        while pos < len(line):
            m = _RECORD.match(line, pos)
            if m is None:
                problems.append(f"line {number}: unreadable record at column {pos}")
                break
            x1, y1, x2, y2 = (int(m[i]) for i in range(2, 6))
            conf = m[8] if m[8] is not None else m[9]
            found[(frame, int(m[1]))] = _Pending(
                frame=frame,
                track_id=int(m[1]),
                box=(x1, y1, x2 + 1, y2 + 1),
                confidence=float(conf) if conf is not None else None,
                encoding=_ENCODING[m[12][0]],
                payload=m[12],
                mask_w=int(m[11]),
                mask_h=int(m[10]),
            )
            pos = m.end() + (1 if m.end() < len(line) and line[m.end()] == "," else 0)
    return found, problems


def _box_size(raw: str) -> tuple[int | None, int | None]:
    match = _SIZE.search(raw)
    return (int(match[1]), int(match[2])) if match else (None, None)


def parse_output(
    raw: str,
    media: str = "video",
    zone_rasters: Mapping[str, np.ndarray] | None = None,
    workers: int | None = None,
    strict: bool = False,
) -> ParsedOutput:
    """Parse raw SAM output into detections sorted by (frame, track id).

    Boxes are ``(left, top, right, bottom)`` with right and bottom exclusive, as the
    reference parser reports them (the wire format is inclusive).

    ``strict=True`` reads the text with Meta's reference parser, which validates every
    mask as it goes (slow; for audits). The default reads the documented grammar
    directly; both give the same detections.

    When a mask for the same frame and object arrives more than once (later
    revisions from prompt propagation), the last one wins. Mask decoding is compiled (see
    :mod:`kappa.sam.fastmask`); ``workers`` > 1 fans it out across processes.
    """
    if strict:
        result = _records(raw, media)
        boxes: dict[tuple[int, int], SegmentationBoxRecord] = {}
        masks: dict[tuple[int, int], SegmentationMaskRecord] = {}
        for record in result.records:
            if isinstance(record, (SegmentationBoxRecord, SegmentationMaskRecord)):
                frame = record.frame.frame_index if record.frame is not None else 0
                key = (frame, int(record.object_id))
                (boxes if isinstance(record, SegmentationBoxRecord) else masks)[key] = record
        pending = [
            _Pending(
                frame=key[0],
                track_id=key[1],
                box=(int(boxes[key].left), int(boxes[key].top), int(boxes[key].right), int(boxes[key].bottom)),
                confidence=boxes[key].confidence if boxes[key].confidence is not None else masks[key].confidence,
                encoding=masks[key].mask.encoding,
                payload=masks[key].mask.payload,
                mask_w=int(masks[key].mask.width),
                mask_h=int(masks[key].mask.height),
            )
            for key in sorted(boxes.keys() & masks.keys())
        ]
        diagnostics: tuple = tuple(result.diagnostics)
    else:
        found, problems = _pending_fast(raw)
        pending = [found[key] for key in sorted(found)]
        diagnostics = tuple(problems)

    items = [(p.encoding, p.payload, p.mask_w, p.mask_h) for p in pending]
    if workers is None:
        workers = 1  # the compiled decoder is faster than process start-up on Windows
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
    return ParsedOutput(detections, width, height, diagnostics)
