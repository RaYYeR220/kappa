"""The Model API's SAM 3.1 wire format, on the service side: prompts in, lines out.

Pure Python + numpy (and OpenCV for one distance transform), so it is unit-tested
locally without torch. The GPU service (``app.py``) ships this file next to the
engine's compiled ``one_bit`` encoder.

Output, one line per frame (documented in Meta's "Read segmentation output")::

    <Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H;c=C|><|mask;x=0;y=0;c=C;data=h,w,!...|>,...

Boxes are inclusive; the mask is cropped to the box and ``one_bit``-encoded. A frame
with no object emits no line. Each line ends with ``\\n``.

Object prompts, the same grammar as the output (documented in "Segmenting with prompts")::

    <30f>1<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|>,2<|point;x=..;y=..;w=..;h=..|>-<|point;...|>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

try:  # inside the Modal container the encoder ships as a top-level module
    from fastencode import encode_mask
except ImportError:  # pragma: no cover - local engine checkout
    from kappa.sam.fastencode import encode_mask

MAX_OBJECTS = 16
DIRECTIONS = ("both", "forward", "backward")

_FRAME = re.compile(r"<(\d+)f>")
_ID = re.compile(r"(\d+)")
_TOKEN = re.compile(r"(-?)<\|(box|point|mask);([^|]*)\|>")


class PromptError(ValueError):
    """A prompt the Model API would answer with a 400."""


@dataclass
class ObjectPrompt:
    """One object's prompt on one frame, in the video's own pixels."""

    frame: int
    obj_id: int
    box: tuple[float, float, float, float] | None = None  # inclusive x1, y1, x2, y2
    points: list[tuple[float, float, int]] = field(default_factory=list)  # x, y, label 1/0

    def sam_inputs(self, width: int, height: int) -> tuple[list[list[float]], list[int]]:
        """Relative point coordinates and SAM labels (box corners first as labels 2 and 3).

        Pixel ``(x, y)`` is the square ``[x, x+1) x [y, y+1)``: a point sits at the pixel
        centre and an inclusive box spans to ``x2 + 1``.
        """
        coords: list[list[float]] = []
        labels: list[int] = []
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            coords += [[x1 / width, y1 / height], [(x2 + 1) / width, (y2 + 1) / height]]
            labels += [2, 3]
        for x, y, label in self.points:
            coords.append([(x + 0.5) / width, (y + 0.5) / height])
            labels.append(label)
        return coords, labels


def is_object_prompt(text: str) -> bool:
    return _FRAME.match(text.strip()) is not None


def _fields(body: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in body.split(";"):
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise PromptError(f"invalid prompt: malformed field {part!r}")
        try:
            out[key.strip()] = float(value)
        except ValueError as error:
            raise PromptError(f"invalid prompt: {key}={value!r} is not a number") from error
    return out


def parse_object_prompts(text: str, width: int, height: int) -> list[ObjectPrompt]:
    """Parse box and point prompts; coordinates are rescaled from the tokens' ``w``/``h``."""
    s = text.strip()
    prompts: list[ObjectPrompt] = []
    frame: int | None = None
    seen: set[int] = set()
    sizes: set[tuple[float, float]] = set()
    pos = 0
    while pos < len(s):
        if s[pos].isspace() or s[pos] == ",":
            pos += 1
            continue
        if m := _FRAME.match(s, pos):
            frame, seen, pos = int(m[1]), set(), m.end()
            continue
        if frame is None:
            raise PromptError("invalid prompt: object prompts must start with a <Nf> frame marker")
        m = _ID.match(s, pos)
        if m is None:
            raise PromptError(f"invalid prompt: expected an object id at column {pos}")
        obj_id, pos = int(m[1]), m.end()
        if obj_id in seen:
            raise PromptError(f"invalid prompt: object id {obj_id} is used more than once on frame {frame}")
        seen.add(obj_id)
        prompt = ObjectPrompt(frame=frame, obj_id=obj_id)
        while (t := _TOKEN.match(s, pos)) is not None:
            negative, kind, f = t[1] == "-", t[2], _fields(t[3])
            pos = t.end()
            if kind == "mask":
                raise PromptError("invalid prompt: mask tokens cannot be used as prompts")
            if "w" not in f or "h" not in f:
                raise PromptError("invalid prompt: every prompt token must declare w and h")
            sizes.add((f["w"], f["h"]))
            sx, sy = width / f["w"], height / f["h"]
            if kind == "box":
                if negative:
                    raise PromptError("invalid prompt: a box names the object; it cannot be negative")
                if prompt.box is not None:
                    raise PromptError(f"invalid prompt: object {obj_id} carries more than one box")
                prompt.box = (f["x1"] * sx, f["y1"] * sy, f["x2"] * sx, f["y2"] * sy)
            else:
                x, y = f["x"] * sx, f["y"] * sy
                if not (0 <= x <= width and 0 <= y <= height):
                    raise PromptError(f"invalid prompt: point ({x}, {y}) lies outside the {width}x{height} frame")
                prompt.points.append((x, y, 0 if negative else 1))
        if prompt.box is None and not prompt.points:
            raise PromptError(f"invalid prompt: object {obj_id} on frame {frame} has no box or point")
        prompts.append(prompt)
    if len(sizes) > 1:
        got = ", ".join(f"{int(w)}x{int(h)}" for w, h in sorted(sizes))
        raise PromptError(f"invalid prompt: every prompt token must declare the same w/h; got {got}")
    for w, h in sizes:
        if abs(w / h - width / height) > 0.01 * width / height:
            raise PromptError(
                f"prompt coordinates are given in a {int(w)}x{int(h)} frame, whose aspect ratio does not "
                f"match the {width}x{height} media"
            )
    if not prompts:
        raise PromptError("invalid prompt: no objects")
    if (n := len({p.obj_id for p in prompts})) > MAX_OBJECTS:
        raise PromptError(f"invalid prompt: {n} objects exceed this model's limit of {MAX_OBJECTS}")
    return prompts


def bf16(value: float) -> float:
    """Round to bfloat16 (round to nearest even), the precision the model scores in."""
    bits = int(np.array(value, dtype=np.float32).view(np.uint32))
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return float(np.array(bits, dtype=np.uint32).view(np.float32))


def format_confidence(value: float) -> str:
    """A DECIMAL per the grammar (digits, optional fraction, never an exponent)."""
    v = bf16(min(max(float(value), 0.0), 1.0))
    text = f"{v:.10f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive (x1, y1, x2, y2) of the mask's pixels, or None for an empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    y1, y2 = int(rows[0]), int(rows[-1])
    cols = np.flatnonzero(mask[y1 : y2 + 1].any(axis=0))
    return int(cols[0]), y1, int(cols[-1]), y2


def format_record(obj_id: int, mask: np.ndarray, confidence: float, width: int, height: int) -> str | None:
    """One ``id<|box|><|mask|>`` record for a full-frame boolean mask (None when empty)."""
    box = mask_box(mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = mask[y1 : y2 + 1, x1 : x2 + 1]
    c = format_confidence(confidence)
    return (
        f"{int(obj_id)}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height};c={c}|>"
        f"<|mask;x=0;y=0;c={c};data={crop.shape[0]},{crop.shape[1]},{encode_mask(crop)}|>"
    )


def format_line(frame: int, ids, masks, confidences, width: int, height: int) -> str | None:
    """A frame's line (records sorted by id, trailing newline), or None if nothing is visible."""
    records = []
    for i in np.argsort(np.asarray(ids), kind="stable"):
        r = format_record(int(ids[i]), masks[i], float(confidences[i]), width, height)
        if r is not None:
            records.append(r)
    return f"<{frame}f>" + ",".join(records) + "\n" if records else None


def match_ids(previous: dict, pending: list, min_iou: float = 0.3, min_frames: int = 3) -> dict[int, int]:
    """Map a new chunk's object ids onto the previous chunk's, from the frames they share.

    ``previous`` maps a position to (ids, masks) the previous chunk emitted there;
    ``pending`` lists (position, ids, masks) of the new chunk on the same positions.
    A pair is scored by its mean mask IoU over the frames where both are present (a new
    chunk confirms an object a few frames late, so absences are not held against it),
    if that is at least ``min_frames`` frames; pairs are assigned one-to-one (Hungarian)
    and a pair below ``min_iou`` stays unmatched. Returns ``{new id: previous id}``.
    """
    from scipy.optimize import linear_sum_assignment

    shared = [(pos, ids, masks) for pos, ids, masks in pending if pos in previous]
    if not shared:
        return {}
    old_ids = sorted({int(i) for pos, _, _ in shared for i in previous[pos][0]})
    new_ids = sorted({int(i) for _, ids, _ in shared for i in ids})
    if not old_ids or not new_ids:
        return {}
    oi = {v: k for k, v in enumerate(old_ids)}
    ni = {v: k for k, v in enumerate(new_ids)}
    total = np.zeros((len(old_ids), len(new_ids)))
    both = np.zeros_like(total)
    for pos, ids, masks in shared:
        pids, pmasks = previous[pos]
        parea = pmasks.reshape(len(pids), -1).sum(axis=1)
        for i, nid in enumerate(ids):
            m = masks[i]
            area = m.sum()
            for j, oid in enumerate(pids):
                r, c = oi[int(oid)], ni[int(nid)]
                both[r, c] += 1
                inter = np.logical_and(m, pmasks[j]).sum()
                if inter:
                    total[r, c] += inter / (area + parea[j] - inter)
    score = np.where(both >= min_frames, total / np.maximum(both, 1), 0.0)
    rows, cols = linear_sum_assignment(score, maximize=True)
    return {new_ids[c]: old_ids[r] for r, c in zip(rows, cols) if score[r, c] >= min_iou}


def carry_prompt(frame: int, obj_id: int, mask: np.ndarray) -> ObjectPrompt | None:
    """A box plus one positive point at the mask's most interior pixel.

    Used to hand an object from one chunk to the next under the same id. The box alone
    is ambiguous when two animals touch (their boxes overlap); the interior point says
    which body inside the box is meant.
    """
    import cv2

    box = mask_box(mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = np.pad(mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8), 1)
    dist = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
    py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
    point = (float(x1 + px - 1), float(y1 + py - 1), 1)
    return ObjectPrompt(frame=frame, obj_id=obj_id, box=(x1, y1, x2, y2), points=[point])
