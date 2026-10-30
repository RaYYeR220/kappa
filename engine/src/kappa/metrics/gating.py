"""Turning a computed number into a reported one, or into a withheld or refused one.

Metric code only says what the data give (a ``Measurement``) or that the data cannot
give anything (``Unmeasurable``). Whether that number may be shown is decided here,
in one place, from the integrity events alone.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kappa.model import Evidence, IntegrityEvent, MetricResult, Span, Status, open_blockers

EVIDENCE_CAP = 50


@dataclass(frozen=True)
class Measurement:
    """A number the data support, not yet judged defensible."""

    value: float
    evidence: tuple[Evidence, ...] = ()
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Unmeasurable:
    """The data cannot yield this number at all; ``reason`` says why in plain English."""

    reason: str


Outcome = Measurement | Unmeasurable


def capped(evidence: Sequence[Evidence], limit: int = EVIDENCE_CAP) -> tuple[Evidence, ...]:
    """At most ``limit`` items, the last one saying how many more stand behind it."""
    if len(evidence) <= limit:
        return tuple(evidence)
    last = evidence[limit - 1]
    more = Evidence(last.span, last.subject, f"{last.note} (+{len(evidence) - limit} more)")
    return (*evidence[: limit - 1], more)


def gate(
    metric_id: str,
    unit: str,
    subject: int | None,
    bin: Span | None,
    window: Span,
    outcome: Outcome,
    events: Iterable[IntegrityEvent],
    depends_on: Iterable[int] | None,
    params: Mapping[str, Any],
) -> MetricResult:
    """Decide the status of one number.

    Refusal comes first: a number that cannot be computed has nothing to unlock, so
    resolving an event would not help. Otherwise any unresolved blocking event on the
    subjects the number depends on, within its window, withholds it.
    """
    head = (metric_id, subject, bin, unit)
    if isinstance(outcome, Measurement) and not math.isfinite(outcome.value):
        outcome = Unmeasurable("the measurement could not be computed from the available frames")
    if isinstance(outcome, Unmeasurable):
        return MetricResult(*head, Status.REFUSED, reasons=(outcome.reason,), params={**params})
    detail: dict[str, Any] = {
        "uncertainty": {**outcome.uncertainty},
        "evidence": outcome.evidence,
        "params": {**params, **outcome.params},
    }
    blockers = open_blockers(events, depends_on, window)
    if blockers:
        reasons = tuple(dict.fromkeys(e.id for e in blockers))
        return MetricResult(
            *head, Status.LOCKED, provisional=outcome.value, reasons=reasons, **detail
        )
    return MetricResult(*head, Status.VERIFIED, value=outcome.value, **detail)
