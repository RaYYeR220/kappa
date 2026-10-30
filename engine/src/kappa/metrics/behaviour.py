"""Counts and durations of detected behaviour bouts.

Overlapping detections of the same behaviour by the same subject are one bout, so time
is never counted twice. Bouts are clipped to the analysed span first; a bout counts in
the window where its clipped start falls, so one already under way when the span opens
counts once, in the first bin, and binned counts add up to the whole.
"""

from __future__ import annotations

from collections.abc import Iterable

from kappa.metrics.gating import Measurement, capped
from kappa.model import BehaviourEvent, Evidence, Span


def bouts(
    events: Iterable[BehaviourEvent], behaviour: str, subject: int, analysed: Span
) -> list[Span]:
    spans = sorted(
        c
        for e in events
        if e.behaviour == behaviour and e.subject == subject and (c := e.span.intersect(analysed))
    )
    merged: list[Span] = []
    for s in spans:
        if merged and s.start < merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, s.end))
        else:
            merged.append(s)
    return merged


def behaviour_count(
    found: list[Span], window: Span, behaviour: str, subject: int, fps: float
) -> Measurement:
    starting = [b for b in found if window.contains(b.start)]
    return Measurement(
        value=float(len(starting)),
        evidence=capped(
            [Evidence(b, subject, f"{behaviour} for {b.length / fps:.1f} s") for b in starting]
        ),
    )


def behaviour_duration(
    found: list[Span], window: Span, behaviour: str, subject: int, fps: float
) -> Measurement:
    parts = [c for b in found if (c := b.intersect(window))]
    return Measurement(
        value=sum(c.length for c in parts) / fps,
        evidence=capped(
            [Evidence(c, subject, f"{behaviour} for {c.length / fps:.1f} s") for c in parts]
        ),
    )
