"""Instrument-versus-human agreement reports: the graded evaluation behind the name.

For one behaviour in one video, :func:`compare_to_raters` scores the instrument
against every human rater *and* every rater against every other rater with the same
statistics on the same frames. The rater-vs-rater numbers are the human ceiling: an
instrument is "within the human range" when it agrees with the raters at least as
well as the least-agreeing pair of humans agree with each other. :func:`aggregate`
does the same over a dataset — pooled frame-level κ, and per-video counts/durations
correlated against the rater mean, with each human held out against the others.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from kappa.eval.stats import (
    Bout,
    bout_match,
    bout_spans,
    cohen_kappa_from_table,
    confusion_table,
    f1_from_counts,
    icc,
    pearson_r,
)
from kappa.model import BehaviourEvent, Span

INSTRUMENT = "instrument"
QUANTITIES = ("count", "duration_s")
Quantity = Literal["count", "duration_s"]
Table = tuple[tuple[int, int], tuple[int, int]]


def _clean(value: Any) -> Any:
    """JSON-safe copy: NaN/inf become None, tuples become lists, keys become strings."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating, np.integer)):
        return _clean(value.item())
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def bouts_from_events(
    events: Iterable[BehaviourEvent], behaviour: str, fps: float, subject: int | None = None
) -> list[Bout]:
    """The instrument's bouts of one behaviour as (onset_s, offset_s), in input order.

    Frame span ``[s, e)`` becomes ``(s/fps, e/fps)``, which the midpoint rule of
    :func:`kappa.eval.stats.bouts_to_frames` maps back onto exactly the same frames.
    """
    return [
        (e.span.start / fps, e.span.end / fps)
        for e in events
        if e.behaviour == behaviour and (subject is None or e.subject == subject)
    ]


# ---------------------------------------------------------------- one video


@dataclass(frozen=True)
class PairAgreement:
    """Agreement between two scorings of one behaviour over one analysis window.

    ``table[i][j]`` counts frames where ``a`` said ``i`` and ``b`` said ``j`` (0 = absent,
    1 = present); it is kept so κ can be pooled across videos exactly. Bout matching
    treats ``a`` as the prediction and ``b`` as the reference (F1 is symmetric).
    Counts are bouts with at least one frame in the window; durations are the union of
    on-frames in seconds, so a rater's overlapping bouts are not double-counted.
    """

    a: str
    b: str
    kappa: float
    table: Table
    bout_tp: int
    bout_fp: int
    bout_fn: int
    count_a: int
    count_b: int
    duration_a_s: float
    duration_b_s: float

    @property
    def bout_f1(self) -> float:
        return f1_from_counts(self.bout_tp, self.bout_fp, self.bout_fn)

    def to_dict(self) -> dict[str, Any]:
        return _clean(
            {
                "a": self.a,
                "b": self.b,
                "kappa": self.kappa,
                "bout_f1": self.bout_f1,
                "bout_tp": self.bout_tp,
                "bout_fp": self.bout_fp,
                "bout_fn": self.bout_fn,
                "count_a": self.count_a,
                "count_b": self.count_b,
                "duration_a_s": self.duration_a_s,
                "duration_b_s": self.duration_b_s,
                "table": self.table,
            }
        )


@dataclass(frozen=True)
class _Scoring:
    spans: list[Span]
    mask: np.ndarray
    fps: float

    @property
    def count(self) -> int:
        return len(self.spans)

    @property
    def duration_s(self) -> float:
        return float(self.mask.sum()) / self.fps


def _score(bouts: Sequence[Bout], fps: float, n_frames: int, offset_s: float) -> _Scoring:
    spans = bout_spans(bouts, fps, n_frames, offset_s)
    mask = np.zeros(n_frames, dtype=bool)
    for s in spans:
        mask[s.start : s.end] = True
    return _Scoring(spans, mask, fps)


def _pair(name_a: str, a: _Scoring, name_b: str, b: _Scoring, min_iou: float) -> PairAgreement:
    t = confusion_table(a.mask, b.mask, labels=[False, True])
    tp, fp, fn, _ = bout_match(a.spans, b.spans, min_iou)
    return PairAgreement(
        a=name_a,
        b=name_b,
        kappa=cohen_kappa_from_table(t),
        table=((int(t[0, 0]), int(t[0, 1])), (int(t[1, 0]), int(t[1, 1]))),
        bout_tp=tp,
        bout_fp=fp,
        bout_fn=fn,
        count_a=a.count,
        count_b=b.count,
        duration_a_s=a.duration_s,
        duration_b_s=b.duration_s,
    )


def _human_pairs(scored: Mapping[str, _Scoring], min_iou: float) -> tuple[PairAgreement, ...]:
    names = sorted(scored)
    return tuple(
        _pair(a, scored[a], b, scored[b], min_iou) for a, b in itertools.combinations(names, 2)
    )


def _check_frames(n_frames: int) -> None:
    if n_frames <= 0:
        raise ValueError(f"need at least one frame to compare on, got n_frames={n_frames}")


def pairwise_agreement(
    raters: Mapping[str, Sequence[Bout]],
    fps: float,
    n_frames: int,
    offset_s: float = 0.0,
    min_iou: float = 0.3,
) -> tuple[PairAgreement, ...]:
    """Every rater against every other (sorted names, each unordered pair once)."""
    _check_frames(n_frames)
    scored = {name: _score(b, fps, n_frames, offset_s) for name, b in raters.items()}
    return _human_pairs(scored, min_iou)


@dataclass(frozen=True)
class AgreementReport:
    """The instrument against each rater, beside the raters against each other.

    ``human_range`` is (min, max) pairwise human κ — None with fewer than two raters,
    in which case ``within_human_range`` is None too: there is no human ceiling to
    compare against, and claiming one would overstate the evidence.
    """

    fps: float
    n_frames: int
    offset_s: float
    min_iou: float
    instrument_count: int
    instrument_duration_s: float
    rater_counts: Mapping[str, int]
    rater_durations_s: Mapping[str, float]
    vs_raters: tuple[PairAgreement, ...]
    between_raters: tuple[PairAgreement, ...]

    @property
    def human_range(self) -> tuple[float, float] | None:
        if not self.between_raters:
            return None
        ks = [p.kappa for p in self.between_raters]
        return (min(ks), max(ks))

    @property
    def instrument_mean_kappa(self) -> float:
        return float(np.mean([p.kappa for p in self.vs_raters]))

    @property
    def within_human_range(self) -> bool | None:
        rng = self.human_range
        return None if rng is None else bool(self.instrument_mean_kappa >= rng[0])

    def quantity(self, who: str, quantity: Quantity) -> float:
        """Count or duration for ``"instrument"`` or a rater name."""
        if quantity == "count":
            return self.instrument_count if who == INSTRUMENT else self.rater_counts[who]
        if quantity == "duration_s":
            return self.instrument_duration_s if who == INSTRUMENT else self.rater_durations_s[who]
        raise ValueError(f"unknown quantity {quantity!r}")

    def to_dict(self) -> dict[str, Any]:
        return _clean(
            {
                "fps": self.fps,
                "n_frames": self.n_frames,
                "offset_s": self.offset_s,
                "min_iou": self.min_iou,
                "instrument_count": self.instrument_count,
                "instrument_duration_s": self.instrument_duration_s,
                "rater_counts": self.rater_counts,
                "rater_durations_s": self.rater_durations_s,
                "instrument_mean_kappa": self.instrument_mean_kappa,
                "human_range": self.human_range,
                "within_human_range": self.within_human_range,
                "vs_raters": [p.to_dict() for p in self.vs_raters],
                "between_raters": [p.to_dict() for p in self.between_raters],
            }
        )


def compare_to_raters(
    instrument: Sequence[Bout],
    raters: Mapping[str, Sequence[Bout]],
    fps: float,
    n_frames: int,
    offset_s: float = 0.0,
    min_iou: float = 0.3,
) -> AgreementReport:
    """Grade the instrument's bouts of one behaviour against human raters on one video.

    All bouts are (onset_s, offset_s) on the raters' clock; frame ``f`` of the analysis
    window starts at ``offset_s + f/fps`` and the window holds ``n_frames`` frames
    (see :func:`kappa.eval.stats.bouts_to_frames` for the rounding rule). Per pair:
    frame-level Cohen's κ, bout F1 at temporal IoU ≥ ``min_iou``, bout counts and
    durations within the window.
    """
    _check_frames(n_frames)
    if not raters:
        raise ValueError("need at least one rater")
    if INSTRUMENT in raters:
        raise ValueError(f"a rater may not be named {INSTRUMENT!r}")
    scored = {name: _score(b, fps, n_frames, offset_s) for name, b in raters.items()}
    inst = _score(instrument, fps, n_frames, offset_s)
    names = sorted(scored)
    return AgreementReport(
        fps=fps,
        n_frames=n_frames,
        offset_s=offset_s,
        min_iou=min_iou,
        instrument_count=inst.count,
        instrument_duration_s=inst.duration_s,
        rater_counts={n: scored[n].count for n in names},
        rater_durations_s={n: scored[n].duration_s for n in names},
        vs_raters=tuple(_pair(INSTRUMENT, inst, n, scored[n], min_iou) for n in names),
        between_raters=_human_pairs(scored, min_iou),
    )


# ---------------------------------------------------------------- dataset


@dataclass(frozen=True)
class PooledPair:
    """One pair's agreement pooled over videos: κ from the summed frame tables (every frame
    of every video weighs the same) and F1 from summed bout TP/FP/FN."""

    a: str
    b: str
    n_videos: int
    kappa: float
    bout_f1: float
    table: Table

    def to_dict(self) -> dict[str, Any]:
        return _clean(self.__dict__)


@dataclass(frozen=True)
class QuantityAgreement:
    """Per-video counts or durations of ``who`` against a reference, across videos.

    The instrument's reference is the mean of all raters on each video; a rater's is the
    mean of the *other* raters (leave-one-out), which is the fair human comparison —
    including the rater in its own reference would inflate its agreement. r and
    ICC(2,1) are NaN with fewer than two videos.
    """

    who: str
    quantity: str
    reference: str
    n_videos: int
    r: float
    icc_2_1: float

    def to_dict(self) -> dict[str, Any]:
        return _clean(self.__dict__)


def _pool(pairs: Sequence[PairAgreement]) -> PooledPair:
    table = np.sum([np.asarray(p.table, dtype=np.int64) for p in pairs], axis=0)
    tp, fp, fn = (sum(getattr(p, f) for p in pairs) for f in ("bout_tp", "bout_fp", "bout_fn"))
    return PooledPair(
        a=pairs[0].a,
        b=pairs[0].b,
        n_videos=len(pairs),
        kappa=cohen_kappa_from_table(table),
        bout_f1=f1_from_counts(tp, fp, fn),
        table=((int(table[0, 0]), int(table[0, 1])), (int(table[1, 0]), int(table[1, 1]))),
    )


def pool_pairs(pairs: Iterable[PairAgreement]) -> tuple[PooledPair, ...]:
    """Pool per-video pair results by (a, b) name; sorted by names."""
    groups: dict[tuple[str, str], list[PairAgreement]] = {}
    for p in pairs:
        groups.setdefault((p.a, p.b), []).append(p)
    return tuple(_pool(groups[k]) for k in sorted(groups))


def _quantity_agreement(
    who: str, quantity: str, reference: str, x: list[float], y: list[float]
) -> QuantityAgreement:
    if len(x) < 2:
        return QuantityAgreement(who, quantity, reference, len(x), math.nan, math.nan)
    return QuantityAgreement(
        who, quantity, reference, len(x), pearson_r(x, y), icc(np.column_stack([x, y]), "2,1")
    )


@dataclass(frozen=True)
class DatasetAgreement:
    """Agreement over many videos of one behaviour."""

    n_videos: int
    instrument_pairs: tuple[PooledPair, ...]
    human_pairs: tuple[PooledPair, ...]
    quantities: tuple[QuantityAgreement, ...]

    @property
    def human_kappa_range(self) -> tuple[float, float] | None:
        if not self.human_pairs:
            return None
        ks = [p.kappa for p in self.human_pairs]
        return (min(ks), max(ks))

    @property
    def instrument_mean_kappa(self) -> float:
        return float(np.mean([p.kappa for p in self.instrument_pairs]))

    @property
    def within_human_range(self) -> bool | None:
        """Pooled κ: the instrument's mean over raters ≥ the lowest human-human κ."""
        rng = self.human_kappa_range
        return None if rng is None else bool(self.instrument_mean_kappa >= rng[0])

    def quantity(self, who: str, quantity: Quantity) -> QuantityAgreement:
        for q in self.quantities:
            if q.who == who and q.quantity == quantity:
                return q
        raise KeyError((who, quantity))

    def human_range(
        self, quantity: Quantity, stat: Literal["r", "icc_2_1"] = "r"
    ) -> tuple[float, float] | None:
        """(min, max) of the raters' leave-one-out agreement on ``quantity``; NaNs ignored."""
        vals = [
            getattr(q, stat)
            for q in self.quantities
            if q.who != INSTRUMENT and q.quantity == quantity
        ]
        vals = [v for v in vals if math.isfinite(v)]
        return (min(vals), max(vals)) if vals else None

    def to_dict(self) -> dict[str, Any]:
        summary = {}
        for qty in QUANTITIES:
            inst = self.quantity(INSTRUMENT, qty)  # type: ignore[arg-type]
            entry: dict[str, Any] = {"instrument": inst.to_dict()}
            for stat in ("r", "icc_2_1"):
                rng = self.human_range(qty, stat)  # type: ignore[arg-type]
                value = getattr(inst, stat)
                entry[f"human_{stat}_range"] = rng
                entry[f"{stat}_within_human_range"] = (
                    None if rng is None or not math.isfinite(value) else bool(value >= rng[0])
                )
            summary[qty] = entry
        return _clean(
            {
                "n_videos": self.n_videos,
                "instrument_mean_kappa": self.instrument_mean_kappa,
                "human_kappa_range": self.human_kappa_range,
                "within_human_range": self.within_human_range,
                "instrument_pairs": [p.to_dict() for p in self.instrument_pairs],
                "human_pairs": [p.to_dict() for p in self.human_pairs],
                "quantities": [q.to_dict() for q in self.quantities],
                "summary": summary,
            }
        )


def aggregate(reports: Sequence[AgreementReport]) -> DatasetAgreement:
    """Dataset-level agreement from per-video reports of the same behaviour.

    Rater names identify the same person across videos; a rater missing from some videos
    is simply left out of those videos' pairs and means.
    """
    if not reports:
        raise ValueError("need at least one video report")
    quantities = []
    for qty in QUANTITIES:
        inst = [float(r.quantity(INSTRUMENT, qty)) for r in reports]
        mean_rater = [float(np.mean([r.quantity(n, qty) for n in r.rater_counts])) for r in reports]
        quantities.append(_quantity_agreement(INSTRUMENT, qty, "mean of raters", inst, mean_rater))
        for name in sorted({n for r in reports for n in r.rater_counts}):
            usable = [r for r in reports if name in r.rater_counts and len(r.rater_counts) > 1]
            own = [float(r.quantity(name, qty)) for r in usable]
            others = [
                float(np.mean([r.quantity(n, qty) for n in r.rater_counts if n != name]))
                for r in usable
            ]
            quantities.append(_quantity_agreement(name, qty, "mean of other raters", own, others))
    return DatasetAgreement(
        n_videos=len(reports),
        instrument_pairs=pool_pairs(p for r in reports for p in r.vs_raters),
        human_pairs=pool_pairs(p for r in reports for p in r.between_raters),
        quantities=tuple(quantities),
    )
