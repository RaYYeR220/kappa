"""What a reader may see of a run: results with locked values withheld, events, tracks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from kappa.model import IntegrityEvent, MetricResult, Status
from kappa.public import public_result


def result_view(r: MetricResult, events: Mapping[str, IntegrityEvent], fps: float) -> dict[str, Any]:
    """A result as the browser may see it.

    Built on :func:`kappa.public.public_result`, plus the evidence spans of a verified
    number so the reader can jump to them. A locked or refused result carries no
    evidence and no uncertainty: a locked time-in-zone is the sum of its evidence
    spans, so sending them would send the withheld number.
    """
    view = public_result(r, events, fps)
    verified = r.status is Status.VERIFIED
    view["evidence"] = (
        [{"span": [e.span.start, e.span.end], "subject": e.subject, "note": e.note} for e in r.evidence]
        if verified
        else []
    )
    view["params"] = dict(r.params)
    return view


def event_view(e) -> dict[str, Any]:
    return {
        "id": e.id,
        "kind": e.kind.value,
        "severity": e.severity.value,
        "subjects": list(e.subjects),
        "span": [e.span.start, e.span.end],
        "detail": e.detail,
        "fix_frame": e.fix_frame,
        "resolved_by": e.resolved_by,
    }


def rounded(a: np.ndarray, digits: int) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in a]
