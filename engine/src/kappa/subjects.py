"""From the segmenter's tracklets to the experiment's subjects.

The segmenter's object ids are only stable while an object stays in view: after an
exit, a full occlusion or a long clip it may report the same animal under a new id,
and it may report objects that are not subjects at all (a reflection, a dropping).
This module decides which tracklets make up which subject, fills dropouts short
enough to bridge honestly, and records every decision a reported number would
otherwise silently depend on as an :class:`IntegrityEvent`.

Everything is deterministic: the same detections, in any order, give the same
subjects, links and event ids.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from kappa.integrity import fold_axis, frames_for, make_event, max_step_px, runs, wrap_axis
from kappa.model import Detection, EventKind, IntegrityEvent, Severity, Span, SubjectTrack, Tracklet
from kappa.protocol import IntegrityPolicy


@dataclass(frozen=True)
class Link:
    """``tracklet_id`` supplied ``subject``'s observed frames within ``span``."""

    subject: int
    tracklet_id: int
    span: Span


@dataclass
class Assembly:
    """Subjects assembled from tracklets, with the evidence of how.

    ``tracks`` are numbered ``1..expected_count``; ``tracklets`` holds every tracklet
    the segmenter reported, linked or not; ``events`` are the integrity events raised
    while assembling (gaps, re-identifications, extra objects).
    """

    tracks: dict[int, SubjectTrack]
    tracklets: dict[int, Tracklet]
    links: list[Link] = field(default_factory=list)
    events: list[IntegrityEvent] = field(default_factory=list)


def _detection_order(d: Detection) -> tuple:
    # A total order, so duplicates and ties resolve the same way whatever order the
    # segmenter's output arrived in.
    conf = -1.0 if d.confidence is None else d.confidence
    return (d.frame, -conf, -d.area_px, d.cx, d.cy, d.major_px, d.minor_px, d.angle_rad)


def group_tracklets(detections: Sequence[Detection]) -> dict[int, Tracklet]:
    """Detections grouped by the segmenter's object id, each in frame order."""
    by_id: dict[int, list[Detection]] = {}
    for d in detections:
        by_id.setdefault(d.track_id, []).append(d)
    return {
        tid: Tracklet(tid, sorted(dets, key=_detection_order))
        for tid, dets in sorted(by_id.items())
    }


def _frames_seen(t: Tracklet) -> int:
    return len({d.frame for d in t.detections})


def _first_frame(t: Tracklet) -> list[Detection]:
    first = t.detections[0].frame
    return list(itertools.takewhile(lambda d: d.frame == first, t.detections))


def _mean_confidence(t: Tracklet) -> float:
    values = [d.confidence for d in t.detections if d.confidence is not None]
    return float(np.mean(values)) if values else 0.0


def _initial_identities(tracklets: dict[int, Tracklet], expected: int) -> list[Tracklet]:
    """The ``expected`` tracklets present earliest, numbered left to right.

    Ties on start frame go to the more confident, then the larger, then the lower id.
    Numbering by x at first sighting gives a convention a person can check on the
    first frame without any tool.
    """
    ranked = sorted(
        tracklets.values(),
        key=lambda t: (
            t.span.start,
            -_mean_confidence(t),
            -float(np.mean([d.area_px for d in t.detections])),
            t.tracklet_id,
        ),
    )
    chosen = ranked[:expected]
    return sorted(chosen, key=lambda t: (t.detections[0].cx, t.detections[0].cy, t.tracklet_id))


@dataclass
class _Chain:
    """A subject being assembled: its track so far and where it was last seen."""

    track: SubjectTrack
    hold_frames: int
    last: int = -1
    prev: int = -1

    @property
    def subject(self) -> int:
        return self.track.subject

    def position(self, frame: int) -> tuple[float, float]:
        return float(self.track.cx[frame]), float(self.track.cy[frame])

    def predict(self, frame: int) -> tuple[float, float] | None:
        """Constant-velocity guess from the last two sightings.

        Across a gap longer than can be interpolated the velocity says nothing, so
        the guess falls back to the last position.
        """
        if self.last < 0:
            return None
        x1, y1 = self.position(self.last)
        if self.prev < 0 or frame - self.last > max(self.hold_frames, 1):
            return x1, y1
        x0, y0 = self.position(self.prev)
        k = (frame - self.last) / (self.last - self.prev)
        return x1 + (x1 - x0) * k, y1 + (y1 - y0) * k


def _put(track: SubjectTrack, frame: int, d: Detection) -> None:
    track.present[frame] = True
    track.cx[frame] = d.cx
    track.cy[frame] = d.cy
    track.area_px[frame] = d.area_px
    track.major_px[frame] = d.major_px
    track.minor_px[frame] = d.minor_px
    track.angle_rad[frame] = d.angle_rad
    track.confidence[frame] = np.nan if d.confidence is None else d.confidence
    track.source[frame] = d.track_id
    for zone, value in d.zone_fraction.items():
        if zone not in track.zone_fraction:
            track.zone_fraction[zone] = np.full(track.n_frames, np.nan)
        track.zone_fraction[zone][frame] = value


def _nearest(dets: list[Detection], guess: tuple[float, float] | None) -> Detection:
    if guess is None:
        return dets[0]
    return min(dets, key=lambda d: math.hypot(d.cx - guess[0], d.cy - guess[1]))


def _write(chain: _Chain, tracklet: Tracklet) -> None:
    """Copy a tracklet into the subject's track.

    Two detections claimed for one frame keep the one nearest where the subject was
    heading: the subject's own motion is the only evidence about which is real.
    """
    for frame, group in itertools.groupby(tracklet.detections, key=lambda d: d.frame):
        dets = list(group)
        det = dets[0] if len(dets) == 1 else _nearest(dets, chain.predict(frame))
        _put(chain.track, frame, det)
        chain.prev, chain.last = chain.last, frame


# -- stitching -------------------------------------------------------------------


def _distance(chain: _Chain, t: Tracklet) -> float:
    x, y = chain.position(chain.last)
    return min(math.hypot(d.cx - x, d.cy - y) for d in _first_frame(t))


def _reachable(chain: _Chain, t: Tracklet, step_px: float) -> bool:
    """Could the subject have got from its last sighting to the tracklet's start?"""
    if chain.last < 0 or t.span.start <= chain.last:
        return False
    return _distance(chain, t) <= step_px * (t.span.start - chain.last) + 1e-9


def _stitch_round(
    chains: list[_Chain], pool: list[Tracklet], step_px: float, window: int
) -> tuple[list[tuple[_Chain, Tracklet]], IntegrityEvent] | None:
    """Link the next re-appearance(s); ``None`` when nothing left can be linked.

    The round starts from the earliest unlinked tracklet some lost subject could
    have reached. Its rivals are the other reachable tracklets that start within the
    interpolation tolerance of it (while it is still in view), and the other lost
    subjects that could have reached any of them. One subject and one tracklet is an
    ordinary re-identification; anything more is a guess, made by motion cost
    (distance from the last sighting) and flagged as blocking.
    """
    first = None
    for t in pool:
        if any(_reachable(c, t, step_px) for c in chains):
            first = t
            break
    if first is None:
        return None

    rivals = [
        t
        for t in pool
        if first.span.start <= t.span.start <= first.span.start + window
        and t.span.start < first.span.end
    ]
    edges = {
        (c.subject, t.tracklet_id) for c in chains for t in rivals if _reachable(c, t, step_px)
    }

    # Connected component of the reachability graph that contains ``first``.
    subjects: set[int] = set()
    tracks: set[int] = {first.tracklet_id}
    while True:
        grown_s = subjects | {s for s, t in edges if t in tracks}
        grown_t = tracks | {t for s, t in edges if s in grown_s}
        if grown_s == subjects and grown_t == tracks:
            break
        subjects, tracks = grown_s, grown_t

    rows = [c for c in chains if c.subject in subjects]
    cols = [t for t in rivals if t.tracklet_id in tracks]
    big = 1e18
    cost = np.array(
        [
            [_distance(c, t) if (c.subject, t.tracklet_id) in edges else big for t in cols]
            for c in rows
        ]
    )
    pairs = [
        (rows[i], cols[j])
        for i, j in zip(*linear_sum_assignment(cost), strict=True)
        if cost[i, j] < big
    ]

    def describe(c: _Chain) -> str:
        options = [
            f"track {t.tracklet_id} (from frame {t.span.start}, {_distance(c, t):.0f} px away)"
            for t in cols
            if (c.subject, t.tracklet_id) in edges
        ]
        return f"subject {c.subject} (last seen frame {c.last}): " + ", ".join(options)

    if len(rows) == 1 and len(cols) == 1:
        ((chain, t),) = pairs
        event = make_event(
            EventKind.ID_CHURN,
            Severity.MINOR,
            (chain.subject,),
            Span(chain.last + 1, t.span.start + 1),
            f"subject {chain.subject} re-identified as track {t.tracklet_id} at frame {t.span.start} "
            f"(last seen frame {chain.last} as track {int(chain.track.source[chain.last])})",
        )
    else:
        choice = "; ".join(f"linked track {t.tracklet_id} to subject {c.subject}" for c, t in pairs)
        event = make_event(
            EventKind.ID_CHURN,
            Severity.BLOCKING,
            tuple(c.subject for c in rows),
            Span(min(c.last for c in rows) + 1, max(t.span.start for _, t in pairs) + 1),
            "ambiguous re-identification; candidates: "
            + "; ".join(describe(c) for c in rows)
            + f". Nearest guess: {choice}. Confirm who is who.",
            fix_frame=min(t.span.start for _, t in pairs),
        )
    return pairs, event


# -- gaps ------------------------------------------------------------------------


def _interpolate(track: SubjectTrack, start: int, end: int) -> None:
    """Fill ``[start, end)`` linearly between the observed frames either side."""
    i, j = start - 1, end
    frac = (np.arange(start, end) - i) / (j - i)
    for name in ("cx", "cy", "area_px", "major_px", "minor_px"):
        arr = getattr(track, name)
        arr[start:end] = arr[i] + (arr[j] - arr[i]) * frac
    # The body axis has no head or tail end: turn the short way round, modulo pi.
    a0 = track.angle_rad[i]
    track.angle_rad[start:end] = fold_axis(a0 + wrap_axis(track.angle_rad[j] - a0) * frac)
    for arr in track.zone_fraction.values():
        arr[start:end] = arr[i] + (arr[j] - arr[i]) * frac
    track.present[start:end] = True
    track.interpolated[start:end] = True


def _fill_gaps(track: SubjectTrack, policy: IntegrityPolicy) -> list[IntegrityEvent]:
    """Bridge short dropouts; flag every absence.

    A dropout no longer than ``max_interpolated_gap_s`` between two sightings is
    filled linearly and noted. Anything longer, and any absence at the start or end
    (there is no second anchor to interpolate towards), blocks if it exceeds the
    tolerance: a subject out of view cannot be measured.
    """
    n, fps, s = track.n_frames, track.fps, track.subject
    tolerance = frames_for(policy.max_interpolated_gap_s, fps)
    events = []
    if not track.present.any():
        return [
            make_event(
                EventKind.GAP,
                Severity.BLOCKING,
                (s,),
                Span(0, n),
                f"subject {s} was never detected",
                fix_frame=0,
            )
        ]
    for start, end in runs(~track.present):
        length = end - start
        seconds = f"{length} frames ({length / fps:.2f} s)"
        interior = start > 0 and end < n
        if interior and length <= tolerance:
            _interpolate(track, start, end)
            events.append(
                make_event(
                    EventKind.GAP,
                    Severity.MINOR,
                    (s,),
                    Span(start, end),
                    f"subject {s} missing for {seconds}; filled by linear interpolation",
                )
            )
            continue
        where = (
            "before its first sighting"
            if start == 0
            else "after its last sighting"
            if end == n
            else "mid-track"
        )
        blocking = length > tolerance
        detail = f"subject {s} not visible for {seconds} {where}"
        if blocking:
            detail += (
                f"; longer than the {policy.max_interpolated_gap_s:.2f} s that may be interpolated"
            )
        events.append(
            make_event(
                EventKind.GAP,
                Severity.BLOCKING if blocking else Severity.MINOR,
                (s,),
                Span(start, end),
                detail,
                fix_frame=start if blocking else None,
            )
        )
    return events


# -- extra objects ---------------------------------------------------------------


def _extras(
    pool: list[Tracklet], tracks: dict[int, SubjectTrack], policy: IntegrityPolicy, fps: float
) -> list[IntegrityEvent]:
    """Scene events for objects that are not any subject.

    Brief detections are segmenter noise. A lasting extra object that never comes
    within a body length of a subject (a reflection, a dropping) is noted; one that
    does could be confused with a subject, and blocks.
    """
    lasting = [t for t in pool if _frames_seen(t) / fps > policy.low_confidence_min_s]
    events = []
    for start, group in itertools.groupby(
        sorted(lasting, key=lambda t: t.span.start), key=lambda t: t.span.start
    ):
        group = list(group)
        near: tuple[int, int, int] | None = None  # (frame, subject, tracklet)
        for t in group:
            for d in t.detections:
                for k, tr in tracks.items():
                    f = d.frame
                    if (
                        tr.present[f]
                        and math.hypot(d.cx - tr.cx[f], d.cy - tr.cy[f]) < tr.major_px[f]
                    ):
                        if near is None or (f, k) < near[:2]:
                            near = (f, k, t.tracklet_id)
                        break
        ids = ", ".join(str(t.tracklet_id) for t in group)
        span = Span(start, max(t.span.end for t in group))
        what = f"track{'s' if len(group) > 1 else ''} {ids} (frames {span.start}-{span.end - 1})"
        if near is None:
            events.append(
                make_event(
                    EventKind.COUNT_MISMATCH,
                    Severity.MINOR,
                    (),
                    span,
                    f"{what} not among the expected subjects; never within a body length of one "
                    "(reflection or debris?)",
                )
            )
        else:
            frame, subject, tid = near
            events.append(
                make_event(
                    EventKind.COUNT_MISMATCH,
                    Severity.BLOCKING,
                    (),
                    span,
                    f"{what} not among the expected subjects; track {tid} comes within a body length of "
                    f"subject {subject} at frame {frame}, so it could be mistaken for it",
                    fix_frame=frame,
                )
            )
    return events


# -- entry point -----------------------------------------------------------------


def assemble(
    detections: Sequence[Detection],
    n_frames: int,
    fps: float,
    expected_count: int,
    policy: IntegrityPolicy,
    px_per_cm: float,
) -> Assembly:
    """Turn raw detections into ``expected_count`` subject tracks plus integrity events.

    Subjects start as the earliest tracklets (numbered left to right). When a
    subject's tracklet ends, a later tracklet it could have reached at
    ``policy.max_speed_cm_s`` continues it. Unlinked lasting tracklets become scene
    events; absences become gap events (short interior ones are interpolated).
    """
    if expected_count < 1:
        raise ValueError("expected_count must be at least 1")
    if n_frames < 1 or fps <= 0 or px_per_cm <= 0:
        raise ValueError("n_frames, fps and px_per_cm must be positive")
    bad = [d for d in detections if not 0 <= d.frame < n_frames]
    if bad:
        raise ValueError(
            f"{len(bad)} detections lie outside frames [0, {n_frames}), e.g. frame {bad[0].frame}"
        )

    tracklets = group_tracklets(detections)
    step_px = max_step_px(policy, px_per_cm, fps)
    window = frames_for(policy.max_interpolated_gap_s, fps)

    chains = [
        _Chain(SubjectTrack.empty(k, n_frames, fps), hold_frames=window)
        for k in range(1, expected_count + 1)
    ]
    links: list[Link] = []
    events: list[IntegrityEvent] = []
    initial = _initial_identities(tracklets, expected_count)
    for chain, t in zip(chains, initial, strict=False):
        _write(chain, t)
        links.append(Link(chain.subject, t.tracklet_id, t.span))

    taken = {t.tracklet_id for t in initial}
    pool = sorted(
        (t for t in tracklets.values() if t.tracklet_id not in taken),
        key=lambda t: (t.span.start, t.tracklet_id),
    )
    while (result := _stitch_round(chains, pool, step_px, window)) is not None:
        pairs, event = result
        events.append(event)
        for chain, t in pairs:
            _write(chain, t)
            links.append(Link(chain.subject, t.tracklet_id, t.span))
        linked = {t.tracklet_id for _, t in pairs}
        pool = [t for t in pool if t.tracklet_id not in linked]

    tracks = {c.subject: c.track for c in chains}
    for track in tracks.values():
        events.extend(_fill_gaps(track, policy))
    events.extend(_extras(pool, tracks, policy, fps))

    links.sort(key=lambda link: (link.subject, link.span.start))
    events.sort(key=lambda e: (e.span.start, e.id))
    return Assembly(tracks=tracks, tracklets=tracklets, links=links, events=events)
