"""Multi-object tracking metrics for identity evaluation (e.g. zebrafish on 3D-ZeF).

Input is one mapping per frame from object id to box, for ground truth and for the
tracker; ids are arbitrary integers and need not agree between the two. Boxes are
``(x_min, y_min, x_max, y_max)`` in pixels. A ground-truth object and a hypothesis can
correspond on a frame only if their boxes overlap with IoU ≥ ``iou_threshold``.

Two complementary views. CLEAR MOT counts per-frame errors, including the moments an
identity changes hands (ID switches). IDF1 asks the question a behavioural measurement
cares about: over the whole video, what fraction of detections carry the *right*
identity — a single long swap costs one switch in MOTA but half the video in IDF1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]
Frames = Sequence[Mapping[int, Box]]


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two axis-aligned boxes; 0 when either has no area."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _check_lengths(gt: Frames, pred: Frames) -> None:
    if len(gt) != len(pred):
        raise ValueError(f"ground truth has {len(gt)} frames but prediction has {len(pred)}")


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def _assign(
    gt: Mapping[int, Box], pred: Mapping[int, Box], gids: list[int], pids: list[int], thr: float
) -> list[tuple[int, int, float]]:
    """Max-IoU one-to-one assignment among pairs with IoU ≥ thr (Hungarian)."""
    if not gids or not pids:
        return []
    iou = np.array([[box_iou(gt[g], pred[p]) for p in pids] for g in gids])
    rows, cols = linear_sum_assignment(np.where(iou >= thr, iou, 0.0), maximize=True)
    return [
        (gids[i], pids[j], float(iou[i, j]))
        for i, j in zip(rows, cols)
        if iou[i, j] >= thr and iou[i, j] > 0
    ]


@dataclass(frozen=True)
class ClearMot:
    """CLEAR MOT totals over a sequence. ``mota``/``motp`` are NaN when undefined."""

    n_frames: int
    n_gt: int
    n_pred: int
    matches: int
    fp: int
    fn: int
    id_switches: int
    mota: float
    motp: float

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in asdict(self).items()
        }


def clear_mot(gt: Frames, pred: Frames, iou_threshold: float = 0.5) -> ClearMot:
    """CLEAR MOT accuracy and precision (Bernardin & Stiefelhagen 2008, EURASIP JIVP).

    Per frame: correspondences from the previous frame are kept while their IoU stays
    ≥ threshold (a valid track is never broken just because another hypothesis fits
    slightly better); the remaining objects and hypotheses are matched by maximum-IoU
    Hungarian assignment. An ID switch is counted whenever an object is matched to a
    different hypothesis than the one it was last matched to, even across a gap.

    MOTA = 1 − (FN + FP + IDSW) / #GT; MOTP = mean IoU over matched pairs (here a
    similarity, so higher is better).
    """
    _check_lengths(gt, pred)
    previous: dict[int, int] = {}  # correspondences on the previous frame
    last_match: dict[int, int] = {}  # most recent hypothesis for each object, ever
    n_gt = n_pred = matches = fp = fn = switches = 0
    iou_sum = 0.0
    for g, p in zip(gt, pred):
        current: dict[int, int] = {}
        for o, h in previous.items():
            if o in g and h in p:
                iou = box_iou(g[o], p[h])
                if iou >= iou_threshold:
                    current[o] = h
                    iou_sum += iou
        used = set(current.values())
        free_g = [o for o in g if o not in current]
        free_p = [h for h in p if h not in used]
        for o, h, iou in _assign(g, p, free_g, free_p, iou_threshold):
            if o in last_match and last_match[o] != h:
                switches += 1
            current[o] = h
            iou_sum += iou
        last_match.update(current)
        previous = current
        n_gt += len(g)
        n_pred += len(p)
        matches += len(current)
        fn += len(g) - len(current)
        fp += len(p) - len(current)
    mota = 1.0 - (fn + fp + switches) / n_gt if n_gt else math.nan
    return ClearMot(
        len(gt), n_gt, n_pred, matches, fp, fn, switches, mota, _ratio(iou_sum, matches)
    )


@dataclass(frozen=True)
class IdentityScores:
    """Identity precision/recall/F1 and the trajectory assignment that achieves them.

    ``assignment`` maps a ground-truth id to the tracker id it was paired with; objects
    paired with no hypothesis are absent. IDP (IDR) is NaN with no predictions (no
    ground truth); IDF1 is 1.0 when both are empty.
    """

    idtp: int
    idfp: int
    idfn: int
    idp: float
    idr: float
    idf1: float
    assignment: Mapping[int, int]

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in asdict(self).items()
            if k != "assignment"
        }
        d["assignment"] = {str(k): v for k, v in self.assignment.items()}
        return d


def idf1(gt: Frames, pred: Frames, iou_threshold: float = 0.5) -> IdentityScores:
    """Identity F1 (Ristani et al. 2016, ECCV workshops, "Performance measures and a data set
    for multi-target, multi-camera tracking").

    Whole trajectories are matched, not frames: each ground-truth trajectory is paired
    with at most one predicted trajectory by a globally optimal bipartite assignment
    that minimises IDFN + IDFP — equivalently maximises IDTP, the number of frames on
    which the paired trajectories overlap with IoU ≥ threshold. Then
    IDFN = #GT detections − IDTP, IDFP = #predicted detections − IDTP,
    IDP = IDTP/(IDTP+IDFP), IDR = IDTP/(IDTP+IDFN), IDF1 = 2·IDTP/(2·IDTP+IDFP+IDFN).
    """
    _check_lengths(gt, pred)
    gids = sorted({o for frame in gt for o in frame})
    pids = sorted({h for frame in pred for h in frame})
    gi = {o: i for i, o in enumerate(gids)}
    pj = {h: j for j, h in enumerate(pids)}
    overlap = np.zeros((len(gids), len(pids)), dtype=np.int64)
    for g, p in zip(gt, pred):
        for o, gb in g.items():
            for h, pb in p.items():
                if box_iou(gb, pb) >= iou_threshold:
                    overlap[gi[o], pj[h]] += 1
    assignment: dict[int, int] = {}
    idtp = 0
    if overlap.size:
        rows, cols = linear_sum_assignment(overlap, maximize=True)
        for i, j in zip(rows, cols):
            if overlap[i, j] > 0:
                assignment[gids[i]] = pids[j]
                idtp += int(overlap[i, j])
    n_gt = sum(len(f) for f in gt)
    n_pred = sum(len(f) for f in pred)
    idfn, idfp = n_gt - idtp, n_pred - idtp
    den = 2 * idtp + idfp + idfn
    return IdentityScores(
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        idp=_ratio(idtp, idtp + idfp),
        idr=_ratio(idtp, idtp + idfn),
        idf1=2 * idtp / den if den else 1.0,
        assignment=assignment,
    )


def id_switches_per_100k_frames(id_switches: int, n_frames: int) -> float:
    """ID switches normalised to 100,000 video frames, so runs of different lengths compare."""
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    return id_switches * 100_000 / n_frames
