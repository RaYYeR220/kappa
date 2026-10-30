"""Agreement statistics for comparing the instrument with human raters.

Human raters score behaviour as bouts in seconds; the instrument works on frames.
Everything here either converts bouts to frames with one documented rule, or
compares two raters' outputs with a standard, cited statistic. Nothing here knows
which side is the instrument: every statistic is computed the same way for
human-vs-human and instrument-vs-human, which is what makes the comparison fair.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment

from kappa.model import Span

Bout = tuple[float, float]
Interval = tuple[float, float] | Span

# ---------------------------------------------------------------- bouts -> frames


def frame_edge(t_s: float, fps: float, offset_s: float = 0.0) -> int:
    """Index of the first frame whose midpoint is at or after ``t_s``.

    This is the single rounding rule used everywhere (see :func:`bouts_to_frames`); a
    window ``[start_s, end_s)`` therefore holds ``frame_edge(end_s, fps, start_s)`` frames.
    The value is rounded to 1e-9 frames before ``ceil`` so that a time sitting exactly on
    a midpoint (e.g. a frame-exact Span converted to seconds) is not pushed to the next
    frame by floating-point noise.
    """
    return math.ceil(round((t_s - offset_s) * fps - 0.5, 9))


def _check_clock(fps: float, n_frames: int) -> None:
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if n_frames < 0:
        raise ValueError(f"n_frames must be non-negative, got {n_frames}")


def bout_spans(
    bouts: Sequence[Bout], fps: float, n_frames: int, offset_s: float = 0.0
) -> list[Span]:
    """Each bout as a half-open frame Span, clipped to ``[0, n_frames)``; empty ones dropped.

    Same rounding rule as :func:`bouts_to_frames`. Bouts keep their identity (overlapping
    bouts stay separate, input order is kept), so counts and bout matching see exactly the
    bouts the rater scored, as resolved at the video's frame rate.
    """
    _check_clock(fps, n_frames)
    spans = []
    for onset, offset in bouts:
        if offset < onset:
            raise ValueError(f"bout ends before it starts: ({onset}, {offset})")
        start = max(frame_edge(onset, fps, offset_s), 0)
        end = min(frame_edge(offset, fps, offset_s), n_frames)
        if end > start:
            spans.append(Span(start, end))
    return spans


def bouts_to_frames(
    bouts: Sequence[Bout], fps: float, n_frames: int, offset_s: float = 0.0
) -> np.ndarray:
    """Boolean per-frame mask of a rater's bouts (onset, offset) in seconds.

    Rounding rule: frame ``f`` covers ``[offset_s + f/fps, offset_s + (f+1)/fps)`` and is
    *on* iff its midpoint ``offset_s + (f + ½)/fps`` lies in the half-open bout
    ``[onset, offset)``; i.e. frames ``ceil(onset·fps − ½) .. ceil(offset·fps − ½) − 1``
    (times relative to ``offset_s``). Midpoint sampling is unbiased (it neither grows nor
    shrinks bouts on average, unlike flooring both ends), maps a frame-exact Span converted
    to seconds back onto itself, and makes a bout shorter than one frame vanish unless it
    covers a midpoint. ``offset_s`` is the label-clock time at which frame 0 starts, for
    comparing on a trimmed analysis window. Bouts are clipped to ``[0, n_frames)``;
    overlapping bouts merge.
    """
    spans = bout_spans(bouts, fps, n_frames, offset_s)
    mask = np.zeros(n_frames, dtype=bool)
    for span in spans:
        mask[span.start : span.end] = True
    return mask


# ---------------------------------------------------------------- Cohen's kappa


def cohen_kappa_from_table(table: ArrayLike) -> float:
    """Cohen's κ from a k×k contingency table (rows: rater A's category, columns: B's).

    κ = (p_o − p_e) / (1 − p_e), with p_o the observed agreement (trace / n) and p_e the
    agreement expected from the two raters' marginals, Σ_k p_A(k)·p_B(k) (Cohen 1960,
    Educ. Psychol. Meas. 20:37). Computed in exact integer arithmetic so that pooling
    tables across videos gives the same κ as concatenating the frames.

    Degenerate case: if p_e = 1 both raters put every item in the same single category,
    κ is 0/0; we return 1.0 when the two ratings are identical and 0.0 otherwise. With
    exact counts p_e = 1 implies identical ratings, so this is 1.0 in practice; the rule
    is stated so that "both raters say the behaviour never happened" reads as agreement.
    """
    t = np.asarray(table)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or t.size == 0:
        raise ValueError(f"need a non-empty square table, got shape {t.shape}")
    if not np.issubdtype(t.dtype, np.integer):
        if not np.all(t == np.round(t)):
            raise ValueError("contingency table must hold counts")
        t = t.astype(np.int64)
    if (t < 0).any():
        raise ValueError("contingency table must hold non-negative counts")
    counts = [[int(v) for v in row] for row in t]
    n = sum(map(sum, counts))
    if n == 0:
        raise ValueError("empty contingency table")
    agree = sum(counts[i][i] for i in range(len(counts)))
    rows = [sum(r) for r in counts]
    cols = [sum(c) for c in zip(*counts)]
    chance = sum(r * c for r, c in zip(rows, cols))  # n² · p_e
    if chance == n * n:
        return 1.0 if agree == n else 0.0
    return (n * agree - chance) / (n * n - chance)


def confusion_table(
    a: ArrayLike, b: ArrayLike, labels: Sequence[object] | None = None
) -> np.ndarray:
    """Contingency counts ``table[i, j]`` = #items rater A put in ``labels[i]`` and B in ``labels[j]``.

    ``labels`` defaults to the sorted union of observed values.
    """
    x, y = np.asarray(a).ravel(), np.asarray(b).ravel()
    if x.shape != y.shape:
        raise ValueError(f"ratings differ in length: {x.size} vs {y.size}")
    if x.size == 0:
        raise ValueError("no items to compare")
    cats = np.unique(np.concatenate([x, y])) if labels is None else np.asarray(labels)
    index = {v: i for i, v in enumerate(cats.tolist())}
    try:
        ia = np.fromiter((index[v] for v in x.tolist()), dtype=np.int64, count=x.size)
        ib = np.fromiter((index[v] for v in y.tolist()), dtype=np.int64, count=y.size)
    except KeyError as exc:
        raise ValueError(f"rating {exc.args[0]!r} is not in labels") from None
    table = np.zeros((len(cats), len(cats)), dtype=np.int64)
    np.add.at(table, (ia, ib), 1)
    return table


def cohen_kappa(a: ArrayLike, b: ArrayLike) -> float:
    """Cohen's κ between two equal-length rating vectors (boolean or categorical).

    Each element is one item (here: one video frame). See :func:`cohen_kappa_from_table`
    for the definition and the degenerate-case rule.
    """
    return cohen_kappa_from_table(confusion_table(a, b))


# ---------------------------------------------------------------- correlation and ICC


def pearson_r(x: ArrayLike, y: ArrayLike) -> float:
    """Pearson product-moment correlation r = cov(x, y) / (σ_x σ_y) (Pearson 1896).

    Returns NaN when either variable is constant: r is undefined there, and reporting 0
    or 1 would invent an answer.
    """
    u, v = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    if u.shape != v.shape:
        raise ValueError(f"x and y differ in length: {u.size} vs {v.size}")
    if u.size < 2:
        raise ValueError("pearson_r needs at least two pairs")
    du, dv = u - u.mean(), v - v.mean()
    den = math.sqrt(float(du @ du) * float(dv @ dv))
    if den == 0.0:
        return math.nan
    return float(du @ dv) / den


def icc(ratings: ArrayLike, kind: Literal["2,1", "3,1"] = "2,1") -> float:
    """Intraclass correlation for ``ratings[n_targets, k_raters]`` (Shrout & Fleiss 1979).

    From the two-way ANOVA mean squares — BMS (between targets), JMS (between judges)
    and EMS (residual) — for a single rater's score:

    - ``"2,1"``: two-way random effects, absolute agreement:
      (BMS − EMS) / (BMS + (k−1)·EMS + k·(JMS − EMS)/n). A rater who is systematically
      high is penalised; use it when the instrument must reproduce the human *numbers*.
    - ``"3,1"``: two-way mixed effects, consistency: (BMS − EMS) / (BMS + (k−1)·EMS).
      Constant offsets between raters are ignored; only the ranking/spacing matters.

    Shrout & Fleiss's Table 2 (6 targets × 4 judges) gives ICC(2,1) = .29, ICC(3,1) = .71.
    Returns NaN when the denominator is zero (no variance at all).
    """
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError(f"need ratings of shape (n_targets >= 2, k_raters >= 2), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("ratings must be finite")
    if kind not in ("2,1", "3,1"):
        raise ValueError(f"unknown ICC kind {kind!r}; expected '2,1' or '3,1'")
    n, k = x.shape
    grand = x.mean()
    ss_targets = k * float(((x.mean(axis=1) - grand) ** 2).sum())
    ss_judges = n * float(((x.mean(axis=0) - grand) ** 2).sum())
    ss_error = float(((x - grand) ** 2).sum()) - ss_targets - ss_judges
    bms = ss_targets / (n - 1)
    jms = ss_judges / (k - 1)
    ems = ss_error / ((n - 1) * (k - 1))
    if kind == "2,1":
        den = bms + (k - 1) * ems + k * (jms - ems) / n
    else:
        den = bms + (k - 1) * ems
    if den == 0.0:
        return math.nan
    return (bms - ems) / den


class BlandAltman(NamedTuple):
    mean_diff: float
    lower_loa: float
    upper_loa: float


def bland_altman(a: ArrayLike, b: ArrayLike) -> BlandAltman:
    """Bias and 95% limits of agreement of ``a − b`` (Bland & Altman 1986, Lancet 1:307).

    mean_diff = d̄, limits = d̄ ± 1.96·s_d with s_d the sample SD (n − 1) of the paired
    differences. Answers "how far apart can two methods be on one sample", which r and
    ICC (scale-free) do not.
    """
    d = np.asarray(a, dtype=float).ravel() - np.asarray(b, dtype=float).ravel()
    if d.size < 2:
        raise ValueError("bland_altman needs at least two pairs")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    return BlandAltman(mean, mean - 1.96 * sd, mean + 1.96 * sd)


# ---------------------------------------------------------------- bout matching


def _bounds(x: Interval) -> tuple[float, float]:
    start, end = (x.start, x.end) if isinstance(x, Span) else (float(x[0]), float(x[1]))
    if end < start:
        raise ValueError(f"interval ends before it starts: ({start}, {end})")
    return start, end


def temporal_iou(a: Interval, b: Interval) -> float:
    """Intersection over union of two half-open time (or frame) intervals; 0 for empty ones."""
    (s1, e1), (s2, e2) = _bounds(a), _bounds(b)
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class BoutMatch:
    pred: int
    ref: int
    iou: float


class BoutMatching(NamedTuple):
    tp: int
    fp: int
    fn: int
    matches: list[BoutMatch]


def bout_match(
    pred: Sequence[Interval], ref: Sequence[Interval], min_iou: float = 0.3
) -> BoutMatching:
    """One-to-one bout matching that maximises total temporal IoU (Hungarian algorithm).

    Pairs below ``min_iou`` are removed before the assignment so they cannot crowd out
    valid pairs; among valid pairs the assignment with the largest summed IoU wins
    (Kuhn 1955 via scipy's linear_sum_assignment). Every matched pair is a true
    positive, unmatched predictions are false positives, unmatched reference bouts are
    misses. Intervals may be ``(start, end)`` tuples or Spans, in any one unit.
    Matches are returned sorted by prediction index.
    """
    p = [_bounds(x) for x in pred]
    r = [_bounds(x) for x in ref]
    if not p or not r:
        return BoutMatching(0, len(p), len(r), [])
    iou = np.array([[temporal_iou(a, b) for b in r] for a in p])
    gated = np.where(iou >= min_iou, iou, 0.0)
    rows, cols = linear_sum_assignment(gated, maximize=True)
    matches = [
        BoutMatch(int(i), int(j), float(iou[i, j]))
        for i, j in zip(rows, cols)
        if iou[i, j] >= min_iou and iou[i, j] > 0
    ]
    tp = len(matches)
    return BoutMatching(tp, len(p) - tp, len(r) - tp, matches)


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """F1 = 2·TP / (2·TP + FP + FN); 1.0 when there is nothing to find and nothing found.

    The empty case mirrors κ's degenerate rule: two raters who both report no bouts agree.
    """
    den = 2 * tp + fp + fn
    return 1.0 if den == 0 else 2 * tp / den


def bout_f1(pred: Sequence[Interval], ref: Sequence[Interval], min_iou: float = 0.3) -> float:
    """Bout-level F1 under :func:`bout_match` (harmonic mean of bout precision and recall)."""
    tp, fp, fn, _ = bout_match(pred, ref, min_iou)
    return f1_from_counts(tp, fp, fn)
