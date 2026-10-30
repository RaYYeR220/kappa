"""What a reader may see of a run: results with locked values withheld, events, tracks."""

from __future__ import annotations

from typing import Any

import numpy as np


def result_view(r) -> dict[str, Any]:
    """A result as the browser may see it: a locked number is never sent."""
    return {
        "metric_id": r.metric_id,
        "subject": r.subject,
        "bin": [r.bin.start, r.bin.end] if r.bin else None,
        "unit": r.unit,
        "status": r.status.value,
        "value": r.value,
        "reasons": list(r.reasons),
        "uncertainty": dict(r.uncertainty),
        "evidence": [{"span": [e.span.start, e.span.end], "subject": e.subject, "note": e.note} for e in r.evidence],
        "params": dict(r.params),
    }


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
