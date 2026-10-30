"""Find the stretches of the subjects' tracks that a number cannot rest on.

Assembly (``kappa.subjects``) flags what goes wrong between tracklets: gaps,
re-identifications, extra objects. This module scans the assembled tracks for what
goes wrong inside them: physically impossible jumps, stretches the segmenter itself
doubted, and contacts between subjects, after which the segmenter's ids may have
crossed. Contacts are settled by motion continuity when the motion is clear enough,
correcting the tracks if the ids crossed; otherwise they block until a person says
who is who.

Event ids are deterministic (``kind:subjects:start``), so the same input always
yields the same ids, and a repair recorded against an id survives a re-run.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

from kappa.model import EventKind, IntegrityEvent, Severity, Span, SubjectTrack
from kappa.protocol import IntegrityPolicy

if TYPE_CHECKING:
    from kappa.subjects import Assembly, Link

#: How far before and after a contact motion continuity looks: long enough to
#: average out mask jitter, short enough that an animal still moves roughly straight.
CONTINUITY_WINDOW_S = 0.3

RESOLVED_KEEP = "motion-continuity"
RESOLVED_SWAP = "motion-continuity-swap"


# -- shared helpers --------------------------------------------------------------


def event_id(kind: EventKind, subjects: Sequence[int], span: Span) -> str:
    return f"{kind.value}:{'-'.join(map(str, subjects)) or 'scene'}:{span.start}"


def make_event(
    kind: EventKind,
    severity: Severity,
    subjects: Iterable[int],
    span: Span,
    detail: str,
    fix_frame: int | None = None,
    resolved_by: str | None = None,
) -> IntegrityEvent:
    """An event with its subjects sorted and its id derived from them."""
    subjects = tuple(sorted(subjects))
    return IntegrityEvent(
        id=event_id(kind, subjects, span),
        kind=kind,
        severity=severity,
        subjects=subjects,
        span=span,
        detail=detail,
        fix_frame=fix_frame,
        resolved_by=resolved_by,
    )


def frames_for(seconds: float, fps: float) -> int:
    """Whole frames within a duration, robust to float error (0.4 s at 30 fps is 12)."""
    return math.floor(seconds * fps + 1e-9)


def max_step_px(policy: IntegrityPolicy, px_per_cm: float, fps: float) -> float:
    """The furthest a subject can move in one frame, in pixels."""
    return policy.max_speed_cm_s * px_per_cm / fps


def wrap_axis(delta: np.ndarray | float) -> np.ndarray | float:
    """Shortest turn between two body-axis angles (an axis is the same modulo pi)."""
    return (np.asarray(delta) + np.pi / 2) % np.pi - np.pi / 2


def fold_axis(angle: np.ndarray | float) -> np.ndarray | float:
    """An axis angle in ``(-pi/2, pi/2]``, the convention of ``kappa.features``."""
    return np.pi / 2 - (np.pi / 2 - np.asarray(angle)) % np.pi


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` runs of True."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def _observed(t: SubjectTrack) -> np.ndarray:
    return t.present & ~t.interpolated


def _window(fps: float) -> int:
    return max(2, round(CONTINUITY_WINDOW_S * fps))


# -- teleports and doubtful detections -----------------------------------------


def teleports(
    tracks: dict[int, SubjectTrack], policy: IntegrityPolicy, px_per_cm: float
) -> list[IntegrityEvent]:
    """Steps between consecutive sightings faster than any subject can move.

    Interpolated frames are skipped: they are straight lines between the two real
    sightings and would only dilute a jump between them.
    """
    events = []
    for k, t in sorted(tracks.items()):
        seen = np.flatnonzero(_observed(t))
        if seen.size < 2:
            continue
        i, j = seen[:-1], seen[1:]
        step = np.hypot(t.cx[j] - t.cx[i], t.cy[j] - t.cy[i])
        speed = step / px_per_cm / ((j - i) / t.fps)
        for x in np.flatnonzero(speed > policy.max_speed_cm_s):
            a, b, px, v = int(i[x]), int(j[x]), float(step[x]), float(speed[x])
            events.append(
                make_event(
                    EventKind.TELEPORT,
                    Severity.BLOCKING,
                    (k,),
                    Span(a, b + 1),
                    f"subject {k} jumped {px / px_per_cm:.1f} cm between frames {a} and {b} "
                    f"({v:.0f} cm/s, above the {policy.max_speed_cm_s:.0f} cm/s limit): "
                    "a wrong-object jump or an identity swap",
                    fix_frame=b,
                )
            )
    return events


def low_confidence(
    tracks: dict[int, SubjectTrack], policy: IntegrityPolicy
) -> list[IntegrityEvent]:
    """Stretches where the segmenter doubted its own detection.

    A missing confidence (NaN) is not low: some segmenters report none. A run is
    bridged across interpolated frames, which carry no confidence of their own, so a
    stretch of doubtful detections with dropouts is judged as the one stretch it is.
    """
    events = []
    for k, t in sorted(tracks.items()):
        with np.errstate(invalid="ignore"):
            low = _observed(t) & (t.confidence < policy.min_confidence)
        for start, end in runs(low | t.interpolated):
            hits = np.flatnonzero(low[start:end])
            if hits.size == 0:
                continue
            s, e = start + int(hits[0]), start + int(hits[-1]) + 1
            seconds = (e - s) / t.fps
            blocking = e - s >= policy.low_confidence_min_s * t.fps - 1e-9
            events.append(
                make_event(
                    EventKind.LOW_CONFIDENCE,
                    Severity.BLOCKING if blocking else Severity.MINOR,
                    (k,),
                    Span(s, e),
                    f"subject {k} detected with confidence below {policy.min_confidence:.2f} for "
                    f"{e - s} frames ({seconds:.2f} s; {policy.low_confidence_min_s:.2f} s blocks)",
                    fix_frame=s if blocking else None,
                )
            )
    return events


# -- contacts --------------------------------------------------------------------


def contact_episodes(
    a: SubjectTrack, b: SubjectTrack, policy: IntegrityPolicy, merge_frames: int = 0
) -> list[Span]:
    """Runs where two subjects' centres are closer than ``contact_distance`` body lengths.

    Body length is the mean of the two major axes on each frame. Runs separated by
    fewer than ``merge_frames`` frames are one episode: identity cannot be judged
    from a window that is itself mostly contact.
    """
    with np.errstate(invalid="ignore"):
        d = np.hypot(a.cx - b.cx, a.cy - b.cy)
        close = (
            a.present & b.present & (d < policy.contact_distance * (a.major_px + b.major_px) / 2)
        )
    spans: list[Span] = []
    for s, e in runs(close):
        if spans and s - spans[-1].end < merge_frames:
            spans[-1] = Span(spans[-1].start, e)
        else:
            spans.append(Span(s, e))
    return spans


def _motion_before(
    t: SubjectTrack, start: int, w: int
) -> tuple[int, np.ndarray, np.ndarray] | None:
    """Last sighting before ``start`` and the median velocity (px/frame) leading up to it."""
    lo = max(0, start - w - 1)
    seen = np.flatnonzero(_observed(t)[lo:start]) + lo
    if seen.size == 0:
        return None
    pos = np.column_stack([t.cx[seen], t.cy[seen]])
    if seen.size >= 2:
        velocity = np.median(np.diff(pos, axis=0) / np.diff(seen)[:, None], axis=0)
    else:
        velocity = np.zeros(2)
    return int(seen[-1]), pos[-1], velocity


def _crossing_frame(
    a: SubjectTrack, b: SubjectTrack, episode: Span
) -> tuple[int, float, float] | None:
    """The frame within a contact where the segmenter's ids most plausibly crossed.

    Exchanging the tracks from frame ``k`` replaces each subject's step into ``k`` by
    a step from its own position to the other's. Where the ids really crossed, that
    removes a jump from both paths; where they did not, it adds one about as long as
    the gap between the animals. Returns ``(k, change, ordinary)``: the frame where
    exchanging shortens the two steps most, the change in their summed length there
    (negative = smoother), and the length of an ordinary frame's two steps in this
    contact. ``None`` if both are never present on two consecutive contact frames.

    This is the evidence motion extrapolation lacks: a segmenter cannot hand an id
    to a body it never touched without the reported centres jumping. (Swapping at
    the end of the contact instead would also put a body-length jump in both tracks.)
    """
    k = np.arange(max(episode.start, 1), min(episode.end, a.n_frames - 1) + 1)
    k = k[a.present[k - 1] & a.present[k] & b.present[k - 1] & b.present[k]]
    if k.size == 0:
        return None
    kept = np.hypot(a.cx[k - 1] - a.cx[k], a.cy[k - 1] - a.cy[k]) + np.hypot(
        b.cx[k - 1] - b.cx[k], b.cy[k - 1] - b.cy[k]
    )
    exchanged = np.hypot(a.cx[k - 1] - b.cx[k], a.cy[k - 1] - b.cy[k]) + np.hypot(
        b.cx[k - 1] - a.cx[k], b.cy[k - 1] - a.cy[k]
    )
    best = int(np.argmin(exchanged - kept))
    ordinary = max(float(np.median(kept)), 1.0)
    return int(k[best]), float(exchanged[best] - kept[best]), ordinary


def _judge_contact(
    a: SubjectTrack, b: SubjectTrack, episode: Span, policy: IntegrityPolicy, w: int
) -> tuple[IntegrityEvent, int | None]:
    """Decide identities after one contact; returns the event and the swap frame, if any.

    Each subject's pre-contact velocity is carried forward from its last sighting and
    compared with where the two are seen just after: if keeping the ids fits that
    motion ``identity_margin`` times better than swapping them, they are kept; if
    swapping fits that much better, the segmenter's ids crossed and are swapped back.
    Either verdict must agree with the tracks inside the contact (see
    :func:`_crossing_frame`): a swap needs a frame where the ids could have crossed
    without a jump, a keep must not leave a jump that exchanging would remove.
    Anything else is a guess and blocks.
    """
    n, fps = a.n_frames, a.fps
    pair = (a.subject, b.subject)
    s, e = episode.start, episode.end
    span = Span(s, min(e + w, n))
    who = f"subjects {pair[0]} and {pair[1]} in contact for {e - s} frames ({(e - s) / fps:.2f} s)"

    def blocking(why: str) -> tuple[IntegrityEvent, None]:
        return (
            make_event(
                EventKind.CONTACT,
                Severity.BLOCKING,
                pair,
                span,
                f"{who}; {why}. Confirm who is who.",
                e,
            ),
            None,
        )

    if e >= n:
        return (
            make_event(
                EventKind.CONTACT,
                Severity.MINOR,
                pair,
                Span(s, n),
                f"{who}, lasting to the end of the analysed range; no identity to carry forward",
            ),
            None,
        )
    before_a, before_b = _motion_before(a, s, w), _motion_before(b, s, w)
    if before_a is None or before_b is None:
        return blocking(
            "no clear sighting of both just before it, so motion cannot carry identity across"
        )
    post = np.arange(e, min(n, e + w))
    post = post[_observed(a)[post] & _observed(b)[post]]
    if post.size == 0:
        return blocking(
            "no clear sighting of both just after it, so motion cannot carry identity across"
        )

    def predicted(before: tuple[int, np.ndarray, np.ndarray]) -> np.ndarray:
        frame, pos, velocity = before
        return pos + velocity * (post - frame)[:, None]

    pa, pb = predicted(before_a), predicted(before_b)
    oa = np.column_stack([a.cx[post], a.cy[post]])
    ob = np.column_stack([b.cx[post], b.cy[post]])

    def cost(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.hypot(*(p - q).T).sum())

    keep, swap = cost(pa, oa) + cost(pb, ob), cost(pa, ob) + cost(pb, oa)
    margin = policy.identity_margin

    crossing = _crossing_frame(a, b, episode)
    if swap > 0 and swap >= margin * keep:
        ratio = swap / keep if keep > 0 else math.inf
        if crossing is not None and crossing[1] < -crossing[2]:
            return blocking(
                f"afterwards the kept identities fit the motion {ratio:.1f}x better, but at frame "
                f"{crossing[0]} both tracks jump in a way exchanging them would remove "
                f"({-crossing[1]:.0f} px), as if the segmenter's ids crossed there"
            )
        return (
            make_event(
                EventKind.CONTACT,
                Severity.MINOR,
                pair,
                span,
                f"{who}; identities kept: carrying their approach forward fits {ratio:.1f}x better than "
                f"swapping (needs {margin:.1f}x)",
                resolved_by=RESOLVED_KEEP,
            ),
            None,
        )
    if keep > 0 and keep >= margin * swap:
        ratio = keep / swap if swap > 0 else math.inf
        if crossing is None or crossing[1] > crossing[2]:
            jump = "" if crossing is None else f" (at best a {crossing[1]:.0f} px jump)"
            return blocking(
                f"afterwards swapped identities fit the motion {ratio:.1f}x better, but the ids "
                f"cannot have crossed anywhere in the contact without a visible jump{jump}; "
                "an approach-and-turn-back looks like a pass-through to motion alone"
            )
        at = crossing[0]
        return (
            make_event(
                EventKind.CONTACT,
                Severity.MINOR,
                pair,
                span,
                f"{who}; the segmenter's ids crossed: swapping them fits the motion {ratio:.1f}x better "
                f"than keeping them (needs {margin:.1f}x), so the tracks were exchanged from frame {at}",
                resolved_by=RESOLVED_SWAP,
            ),
            at,
        )
    lo, hi = sorted((keep, swap))
    ratio = hi / lo if lo > 0 else 1.0
    return blocking(
        f"afterwards keeping and swapping identities fit the motion within {ratio:.1f}x of each other "
        f"(needs {margin:.1f}x), so motion cannot tell them apart"
    )


_ARRAYS = (
    "present",
    "cx",
    "cy",
    "area_px",
    "major_px",
    "minor_px",
    "angle_rad",
    "confidence",
    "source",
    "interpolated",
)


def _copy(t: SubjectTrack) -> SubjectTrack:
    return replace(
        t,
        **{name: getattr(t, name).copy() for name in _ARRAYS},
        x_cm=None if t.x_cm is None else t.x_cm.copy(),
        y_cm=None if t.y_cm is None else t.y_cm.copy(),
        zone_fraction={z: v.copy() for z, v in t.zone_fraction.items()},
    )


def _exchange(x: np.ndarray, y: np.ndarray, frame: int) -> None:
    tail = x[frame:].copy()
    x[frame:] = y[frame:]
    y[frame:] = tail


def _swap_from(a: SubjectTrack, b: SubjectTrack, frame: int) -> None:
    """Exchange everything two tracks hold from ``frame`` on."""
    for name in _ARRAYS:
        _exchange(getattr(a, name), getattr(b, name), frame)
    for name in ("x_cm", "y_cm"):
        x, y = getattr(a, name), getattr(b, name)
        if x is None and y is None:
            continue
        if x is None or y is None:
            raise ValueError(
                f"cannot swap {name}: only one of subjects {a.subject}, {b.subject} has it"
            )
        _exchange(x, y, frame)
    for zone in sorted(set(a.zone_fraction) | set(b.zone_fraction)):
        x = a.zone_fraction.setdefault(zone, np.full(a.n_frames, np.nan))
        y = b.zone_fraction.setdefault(zone, np.full(b.n_frames, np.nan))
        _exchange(x, y, frame)


def _resolve(
    tracks: dict[int, SubjectTrack], policy: IntegrityPolicy
) -> tuple[dict[int, SubjectTrack], list[IntegrityEvent], list[tuple[int, int, int]]]:
    """Judge every contact in time order, swapping as needed; also returns the swaps.

    A swap is applied to the end of the range and later contacts are judged on the
    corrected tracks, which is the same as swapping only until the pair's next
    contact and deciding afresh there. Swaps change who meets whom later, so the
    episodes are recomputed after each one.
    """
    fixed = {k: _copy(t) for k, t in sorted(tracks.items())}
    keys = sorted(fixed)
    if len(keys) < 2:
        return fixed, [], []
    w = _window(fixed[keys[0]].fps)

    def episodes() -> list[tuple[int, int, int, int]]:
        return sorted(
            (ep.end, ep.start, a, b)
            for a, b in combinations(keys, 2)
            for ep in contact_episodes(fixed[a], fixed[b], policy, merge_frames=w)
        )

    events: list[IntegrityEvent] = []
    swaps: list[tuple[int, int, int]] = []
    pending, last = episodes(), (-1, -1, -1, -1)
    while (nxt := next((x for x in pending if x > last), None)) is not None:
        last = nxt
        end, start, a, b = nxt
        event, at = _judge_contact(fixed[a], fixed[b], Span(start, end), policy, w)
        events.append(event)
        if at is not None:
            _swap_from(fixed[a], fixed[b], at)
            swaps.append((at, a, b))
            pending = episodes()
    return fixed, events, swaps


def resolve_contacts(
    tracks: dict[int, SubjectTrack], policy: IntegrityPolicy, px_per_cm: float
) -> tuple[dict[int, SubjectTrack], list[IntegrityEvent]]:
    """Tracks with crossed ids swapped back, and one CONTACT event per contact.

    The input tracks are not modified. ``px_per_cm`` is unused (contact is measured
    in body lengths) but kept so every scan shares one signature.
    """
    fixed, events, _ = _resolve(tracks, policy)
    return fixed, events


# -- putting it together ---------------------------------------------------------


def _merge(*groups: Iterable[IntegrityEvent]) -> list[IntegrityEvent]:
    """Events from all sources, first occurrence of each id kept, ordered by start."""
    seen: dict[str, IntegrityEvent] = {}
    for group in groups:
        for event in group:
            seen.setdefault(event.id, event)
    return sorted(seen.values(), key=lambda e: (e.span.start, e.id))


def scan(
    tracks: dict[int, SubjectTrack], policy: IntegrityPolicy, px_per_cm: float
) -> list[IntegrityEvent]:
    """CONTACT, TELEPORT and LOW_CONFIDENCE events for the given tracks.

    Teleports and low confidence are judged on the tracks after contact correction,
    so their subjects match what :func:`resolve_contacts` returns. Input not modified.
    """
    fixed, contacts, _ = _resolve(tracks, policy)
    return _merge(contacts, teleports(fixed, policy, px_per_cm), low_confidence(fixed, policy))


def _relabel(
    subjects: tuple[int, ...], start: int, swaps: Sequence[tuple[int, int, int]]
) -> tuple[int, ...]:
    for frame, a, b in swaps:
        if start >= frame:
            subjects = tuple(b if s == a else a if s == b else s for s in subjects)
    return subjects


def _relabel_event(event: IntegrityEvent, swaps: Sequence[tuple[int, int, int]]) -> IntegrityEvent:
    subjects = _relabel(event.subjects, event.span.start, swaps)
    if tuple(sorted(subjects)) == event.subjects:
        return event
    note = f" [raised for subject(s) {', '.join(map(str, event.subjects))} before the post-contact identity correction]"
    return make_event(
        event.kind,
        event.severity,
        subjects,
        event.span,
        event.detail + note,
        event.fix_frame,
        event.resolved_by,
    )


def _relabel_links(links: Sequence[Link], swaps: Sequence[tuple[int, int, int]]) -> list[Link]:
    out = list(links)
    for frame, a, b in swaps:
        other = {a: b, b: a}
        nxt = []
        for link in out:
            if link.subject not in other or link.span.end <= frame:
                nxt.append(link)
            elif link.span.start >= frame:
                nxt.append(replace(link, subject=other[link.subject]))
            else:
                nxt.append(replace(link, span=Span(link.span.start, frame)))
                nxt.append(
                    replace(link, subject=other[link.subject], span=Span(frame, link.span.end))
                )
        out = nxt
    return sorted(out, key=lambda link: (link.subject, link.span.start, link.tracklet_id))


def check(assembly: Assembly, policy: IntegrityPolicy, px_per_cm: float) -> Assembly:
    """The assembly with contacts resolved and every integrity event attached.

    Assembly events and links raised for a subject after a corrected contact are
    relabelled to the subject whose data they now describe. The input is not modified.
    """
    fixed, contacts, swaps = _resolve(assembly.tracks, policy)
    carried = [_relabel_event(e, swaps) for e in assembly.events]
    events = _merge(
        carried, contacts, teleports(fixed, policy, px_per_cm), low_confidence(fixed, policy)
    )
    return replace(
        assembly, tracks=fixed, links=_relabel_links(assembly.links, swaps), events=events
    )


def all_events(
    assembly: Assembly, policy: IntegrityPolicy, px_per_cm: float
) -> list[IntegrityEvent]:
    """Assembly events plus scan events, ordered by span start, one per id.

    The tracks these events describe are ``check(assembly, ...).tracks``.
    """
    return check(assembly, policy, px_per_cm).events


# -- repairs and reporting -------------------------------------------------------


def resolve(events: Sequence[IntegrityEvent], event_id: str, by: str) -> list[IntegrityEvent]:
    """A copy of ``events`` with ``event_id`` marked resolved by ``by`` (e.g. a repair id)."""
    if not by:
        raise ValueError("a resolution needs to say who or what resolved it")
    if not any(e.id == event_id for e in events):
        raise KeyError(event_id)
    return [replace(e, resolved_by=by) if e.id == event_id else e for e in events]


def summary(events: Iterable[IntegrityEvent]) -> dict[str, Any]:
    """Counts for a report: by kind and severity, and which blockers are still open."""
    events = list(events)
    by_kind: dict[str, dict[str, int]] = {}
    for e in events:
        counts = by_kind.setdefault(e.kind.value, {s.value: 0 for s in Severity})
        counts[e.severity.value] += 1
    severity = Counter(e.severity.value for e in events)
    blockers = [e.id for e in events if e.is_open_blocker]
    return {
        "total": len(events),
        "by_kind": dict(sorted(by_kind.items())),
        "by_severity": {s.value: severity.get(s.value, 0) for s in Severity},
        "resolved": sum(e.resolved_by is not None for e in events),
        "open_blockers": len(blockers),
        "open_blocker_ids": blockers,
    }
