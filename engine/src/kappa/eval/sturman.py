"""The Sturman et al. 2020 rater labels: loading, and the human agreement they show.

Three experimenters independently scored the same open-field (and elevated-plus-maze)
videos, as bouts with onset/offset in seconds. Each rater also placed Start/End
markers around the period they scored. That is exactly the human ceiling Kappa is
graded against: how well do trained people agree with *each other* on these videos?

File format (semicolon-separated, quoted or not): ``file;Experimenter;from;to;type;ID;DLCFile``.
``type`` is the behaviour, or a Start/End marker (spelled ``Start/End``, ``StartEnd`` or
``Start_End``), or an annotation-tool placeholder (``_DEFAULT``, ``NA``) that carries no
behaviour and is skipped. One rater may contribute several files per video (one per
annotation pass); they are merged.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kappa.eval.report import (
    AgreementReport,
    PairAgreement,
    compare_to_raters,
    pairwise_agreement,
    pool_pairs,
)
from kappa.eval.stats import Bout, frame_edge, pearson_r

OFT_BEHAVIOURS = ("supported_rear", "unsupported_rear", "grooming", "jumping")
# Jumping is left out of the default agreement set: 16 bouts over 60 rater-videos is too
# rare for a stable κ (one disputed jump moves it by tenths).
SCORED_BEHAVIOURS = ("supported_rear", "unsupported_rear", "grooming")

_ALIASES = {"supported": "supported_rear", "unsupported": "unsupported_rear"}
_MARKERS = frozenset({"start/end", "startend", "start_end"})
_PLACEHOLDERS = frozenset({"_default", "na", ""})
_REQUIRED = ("Experimenter", "from", "to", "type", "ID")

Window = tuple[float, float]


def normalise_behaviour(name: str) -> str | None:
    """Stable snake_case id for a label ``type``; None for markers and placeholders.

    The open-field names map to Kappa's ids (``Supported`` → ``supported_rear``,
    ``Unsupported`` → ``unsupported_rear``); anything else is lower-cased with runs of
    non-alphanumerics turned into ``_`` (``Head Dip`` → ``head_dip``).
    """
    key = name.strip().lower()
    if key in _MARKERS or key in _PLACEHOLDERS:
        return None
    if key in _ALIASES:
        return _ALIASES[key]
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_") or None


def _natural(s: str) -> tuple[Any, ...]:
    # "OFT_5" sorts before "OFT_11".
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class SturmanLabels:
    """``bouts[video][rater][behaviour]`` → bouts sorted by onset; ``windows[video][rater]``
    → the rater's scored period, or None if they did not mark both ends.

    Every rater present on a video has a (possibly empty) list for every behaviour that
    occurs anywhere in the file, so "scored none" and "not scored" are distinguishable:
    the first is an empty list, the second a missing rater.
    """

    bouts: dict[str, dict[str, dict[str, list[Bout]]]]
    windows: dict[str, dict[str, Window | None]]
    n_skipped_rows: int = 0
    behaviours: tuple[str, ...] = field(default=())

    @property
    def videos(self) -> tuple[str, ...]:
        return tuple(sorted(self.bouts, key=_natural))

    @property
    def raters(self) -> tuple[str, ...]:
        return tuple(sorted({r for v in self.bouts.values() for r in v}))

    def common_window(self, video_id: str) -> Window:
        """The period every rater who marked one claims to have scored (their intersection).

        Frames outside it are excluded from agreement: a rater cannot "agree that nothing
        happened" during a stretch they did not watch.
        """
        marked = [w for w in self.windows.get(video_id, {}).values() if w is not None]
        if not marked:
            raise ValueError(f"{video_id}: no rater marked a Start/End window")
        start, end = max(w[0] for w in marked), min(w[1] for w in marked)
        if end <= start:
            raise ValueError(f"{video_id}: raters' windows do not overlap")
        return (start, end)


def load_sturman(path: str | Path) -> SturmanLabels:
    """Parse a Sturman label CSV (``AllLabDataOFT_final.csv`` or ``AllEPMLables.csv``).

    A rater's window is the hull of their markers — earliest marker onset to latest
    marker offset — when they placed at least two; a stray marker mid-video then falls
    inside the hull and is harmless. Rows that are placeholders, have non-numeric times
    or end before they start are skipped and counted in ``n_skipped_rows``.
    """
    bouts: dict[str, dict[str, dict[str, list[Bout]]]] = {}
    markers: dict[str, dict[str, list[Bout]]] = {}
    seen: set[str] = set()
    skipped = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for row in reader:
            video, rater, kind = row["ID"].strip(), row["Experimenter"].strip(), row["type"]
            onset, offset = _number(row["from"]), _number(row["to"])
            behaviour = normalise_behaviour(kind)
            is_marker = kind.strip().lower() in _MARKERS
            if (
                onset is None
                or offset is None
                or offset < onset
                or (behaviour is None and not is_marker)
            ):
                skipped += 1
                continue
            bouts.setdefault(video, {}).setdefault(rater, {})
            if is_marker:
                markers.setdefault(video, {}).setdefault(rater, []).append((onset, offset))
            else:
                bouts[video][rater].setdefault(behaviour, []).append((onset, offset))  # type: ignore[arg-type]
                seen.add(behaviour)  # type: ignore[arg-type]
    for per_video in bouts.values():
        for per_rater in per_video.values():
            for b in seen:
                per_rater.setdefault(b, [])
            for b in per_rater:
                per_rater[b].sort()
    windows: dict[str, dict[str, Window | None]] = {}
    for video, per_video in bouts.items():
        windows[video] = {}
        for rater in per_video:
            m = markers.get(video, {}).get(rater, [])
            windows[video][rater] = (
                (min(a for a, _ in m), max(b for _, b in m)) if len(m) >= 2 else None
            )
    return SturmanLabels(bouts, windows, skipped, tuple(sorted(seen)))


def _window_frames(labels: SturmanLabels, video_id: str, fps: float) -> tuple[float, int]:
    start, end = labels.common_window(video_id)
    return start, frame_edge(end, fps, start)


def _behaviour_bouts(labels: SturmanLabels, video_id: str, behaviour: str) -> dict[str, list[Bout]]:
    if behaviour not in labels.behaviours:
        raise ValueError(f"behaviour {behaviour!r} not in labels; have {labels.behaviours}")
    return {r: b[behaviour] for r, b in labels.bouts[video_id].items()}


def compare_video(
    labels: SturmanLabels,
    video_id: str,
    behaviour: str,
    instrument: Sequence[Bout],
    fps: float = 25.0,
    min_iou: float = 0.3,
) -> AgreementReport:
    """Grade the instrument's bouts on one Sturman video, over the raters' common window.

    ``instrument`` bouts are in seconds of video time (e.g. from
    :func:`kappa.eval.report.bouts_from_events`), the same clock as the labels.
    """
    offset_s, n_frames = _window_frames(labels, video_id, fps)
    raters = _behaviour_bouts(labels, video_id, behaviour)
    return compare_to_raters(instrument, raters, fps, n_frames, offset_s=offset_s, min_iou=min_iou)


@dataclass(frozen=True)
class RaterPair:
    """One pair of raters over the whole dataset, for one behaviour.

    ``kappa`` is frame-level κ pooled over every frame of every video's common window
    (pooling keeps rare behaviours estimable where many videos have no bouts at all);
    ``bout_f1`` pools bout TP/FP/FN the same way; ``count_r``/``duration_r`` are Pearson r
    of the two raters' per-video bout counts / scored seconds.
    """

    a: str
    b: str
    n_videos: int
    kappa: float
    bout_f1: float
    count_r: float
    duration_r: float


@dataclass(frozen=True)
class HumanAgreement:
    fps: float
    min_iou: float
    pairs: Mapping[str, tuple[RaterPair, ...]]

    def _range(self, behaviour: str, stat: str) -> tuple[float, float]:
        vals = [getattr(p, stat) for p in self.pairs[behaviour]]
        return (min(vals), max(vals))

    def kappa_range(self, behaviour: str) -> tuple[float, float]:
        return self._range(behaviour, "kappa")

    def count_r_range(self, behaviour: str) -> tuple[float, float]:
        return self._range(behaviour, "count_r")

    def to_dict(self) -> dict[str, Any]:
        def clean(v: Any) -> Any:
            return None if isinstance(v, float) and not math.isfinite(v) else v

        return {
            "fps": self.fps,
            "min_iou": self.min_iou,
            "behaviours": {
                b: [{k: clean(v) for k, v in p.__dict__.items()} for p in pairs]
                for b, pairs in self.pairs.items()
            },
        }


def _per_video_r(group: Sequence[PairAgreement], attr_a: str, attr_b: str) -> float:
    if len(group) < 2:
        return math.nan
    return pearson_r([getattr(p, attr_a) for p in group], [getattr(p, attr_b) for p in group])


def human_agreement(
    labels: SturmanLabels,
    fps: float = 25.0,
    behaviours: Sequence[str] = SCORED_BEHAVIOURS,
    min_iou: float = 0.3,
) -> HumanAgreement:
    """Pairwise rater agreement per behaviour — the human ceiling on this dataset.

    Each video is analysed over its common window (see :meth:`SturmanLabels.common_window`)
    at ``fps``; bouts become frames by the midpoint rule of
    :func:`kappa.eval.stats.bouts_to_frames`.
    """
    out: dict[str, tuple[RaterPair, ...]] = {}
    for behaviour in behaviours:
        per_video = []
        for video in labels.videos:
            offset_s, n_frames = _window_frames(labels, video, fps)
            raters = _behaviour_bouts(labels, video, behaviour)
            per_video.extend(
                pairwise_agreement(raters, fps, n_frames, offset_s=offset_s, min_iou=min_iou)
            )
        rows = []
        for pooled in pool_pairs(per_video):
            group = [p for p in per_video if (p.a, p.b) == (pooled.a, pooled.b)]
            rows.append(
                RaterPair(
                    a=pooled.a,
                    b=pooled.b,
                    n_videos=pooled.n_videos,
                    kappa=pooled.kappa,
                    bout_f1=pooled.bout_f1,
                    count_r=_per_video_r(group, "count_a", "count_b"),
                    duration_r=_per_video_r(group, "duration_a_s", "duration_b_s"),
                )
            )
        out[behaviour] = tuple(rows)
    return HumanAgreement(fps, min_iou, out)
