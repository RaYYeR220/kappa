"""Time in, entries into and latency to a zone.

A visit is a run of frames on which the subject counts as inside. An unseen frame is
neither inside nor outside, so it ends a run without being an exit: a visit counts as
an entry only if the last frame on which the subject *was* seen had it outside. That
also means being placed in the zone at the start, or first appearing there, is not an
entry. Visits shorter than the entry rule's ``min_dwell_s`` still add to the time in
the zone but are not entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kappa import geometry
from kappa.metrics.frames import SubjectView, frames_for
from kappa.metrics.gating import Measurement, Unmeasurable, capped
from kappa.model import Evidence, Span
from kappa.protocol import AssayProtocol, ZoneSpec

NOT_PROJECTED = "the track has no arena coordinates (it was never projected onto the arena)"


@dataclass(frozen=True, eq=False)
class ZoneVisits:
    view: SubjectView
    zone: ZoneSpec
    inside: np.ndarray
    visits: tuple[Span, ...]
    entries: tuple[Span, ...]


def zone_visits(
    view: SubjectView, protocol: AssayProtocol, zone_id: str
) -> ZoneVisits | Unmeasurable:
    zone = next((z for z in protocol.zones if z.id == zone_id), None)
    if zone is None:
        return Unmeasurable(f"zone {zone_id} is not defined in the protocol")
    rule = protocol.entry_rule
    if rule.mode == "centroid":
        if view.x is None or view.y is None:
            return Unmeasurable(NOT_PROJECTED)
        inside = view.present & geometry.inside(zone, protocol.arena, view.x, view.y)
    elif rule.mode == "body_fraction":
        fraction = view.zone_fraction.get(zone_id)
        if fraction is None:
            return Unmeasurable(
                f"body-fraction data is unavailable for zone {zone_id}; "
                "the entry rule needs the share of the body mask inside it on every frame"
            )
        inside = view.present & (fraction >= rule.body_fraction)
    else:
        return Unmeasurable(f"entry rule {rule.mode} is not supported")

    visits = tuple(view.spans(inside))
    frame = np.arange(inside.shape[0])
    last_seen = np.maximum.accumulate(np.where(view.present, frame, -1))
    need = frames_for(rule.min_dwell_s, view.fps)

    def entered(visit: Span) -> bool:
        before = visit.start - view.span.start - 1
        seen = last_seen[before] if before >= 0 else -1
        return seen >= 0 and not inside[seen] and visit.length >= need

    return ZoneVisits(view, zone, inside, visits, tuple(v for v in visits if entered(v)))


def _visit_evidence(zv: ZoneVisits, window: Span) -> tuple[Evidence, ...]:
    fps, subject = zv.view.fps, zv.view.subject
    parts = [c for v in zv.visits if (c := v.intersect(window))]
    return capped(
        [Evidence(c, subject, f"in {zv.zone.id} for {c.length / fps:.1f} s") for c in parts]
    )


def time_in_zone(zv: ZoneVisits, window: Span) -> Measurement:
    frames = int(zv.inside[zv.view.local(window)].sum())
    return Measurement(value=frames / zv.view.fps, evidence=_visit_evidence(zv, window))


def fraction_time_in_zone(zv: ZoneVisits, window: Span) -> Measurement:
    """Share of the frames the subject was seen on, so unseen time does not dilute it."""
    sl = zv.view.local(window)
    seen = int(zv.view.present[sl].sum())
    frames = int(zv.inside[sl].sum())
    return Measurement(value=100.0 * frames / seen, evidence=_visit_evidence(zv, window))


def _entries_in(zv: ZoneVisits, window: Span) -> list[Span]:
    """Entries count in the window where they begin, so binned counts add up."""
    return [e for e in zv.entries if window.contains(e.start)]


def zone_entries(zv: ZoneVisits, window: Span) -> Measurement:
    fps, subject = zv.view.fps, zv.view.subject
    entries = _entries_in(zv, window)
    return Measurement(
        value=float(len(entries)),
        evidence=capped(
            [
                Evidence(e, subject, f"entered {zv.zone.id}, stayed {e.length / fps:.1f} s")
                for e in entries
            ]
        ),
    )


def latency_to_zone(zv: ZoneVisits, window: Span) -> Measurement:
    """Time to the first entry; never entering is censored at the window's length."""
    fps, subject = zv.view.fps, zv.view.subject
    entries = _entries_in(zv, window)
    if not entries:
        return Measurement(
            value=window.length / fps,
            evidence=(Evidence(window, subject, f"never entered {zv.zone.id} in this span"),),
            params={"censored": True},
        )
    first = entries[0]
    seconds = (first.start - window.start) / fps
    return Measurement(
        value=seconds,
        evidence=(Evidence(first, subject, f"first entry into {zv.zone.id}, {seconds:.1f} s in"),),
        params={"censored": False},
    )
