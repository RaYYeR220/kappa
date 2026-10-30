import numpy as np
import pytest

from kappa.model import (
    EventKind,
    IntegrityEvent,
    MetricResult,
    Severity,
    Span,
    Status,
    SubjectTrack,
    open_blockers,
)


def test_span_rejects_negative_and_reversed():
    with pytest.raises(ValueError):
        Span(-1, 3)
    with pytest.raises(ValueError):
        Span(5, 4)


def test_span_overlap_is_half_open():
    assert Span(0, 10).overlaps(Span(9, 12))
    assert not Span(0, 10).overlaps(Span(10, 12))
    assert Span(0, 10).intersect(Span(5, 20)) == Span(5, 10)
    assert Span(0, 10).intersect(Span(10, 20)) is None


def _event(kind=EventKind.TELEPORT, severity=Severity.BLOCKING, subjects=(1,), span=Span(100, 101), resolved=None):
    return IntegrityEvent(
        id=f"{kind.value}:{subjects}:{span.start}",
        kind=kind,
        severity=severity,
        subjects=subjects,
        span=span,
        detail="",
        resolved_by=resolved,
    )


def test_open_blockers_filters_by_subject_span_severity_and_resolution():
    events = [
        _event(subjects=(1,), span=Span(100, 101)),
        _event(subjects=(2,), span=Span(100, 101)),
        _event(severity=Severity.MINOR, subjects=(1,), span=Span(50, 60)),
        _event(subjects=(1,), span=Span(300, 310), resolved="repair-1"),
        _event(kind=EventKind.COUNT_MISMATCH, subjects=(), span=Span(200, 210)),
    ]
    hits = open_blockers(events, [1], Span(0, 250))
    assert [e.span.start for e in hits] == [100, 200]
    assert open_blockers(events, [1], Span(101, 200)) == []
    assert len(open_blockers(events, None, Span(0, 1000))) == 3


def test_metric_result_enforces_withholding():
    with pytest.raises(ValueError):
        MetricResult("d", 1, None, "cm", Status.VERIFIED)
    with pytest.raises(ValueError):
        MetricResult("d", 1, None, "cm", Status.LOCKED, value=3.0, reasons=("x",))
    with pytest.raises(ValueError):
        MetricResult("d", 1, None, "cm", Status.REFUSED)
    locked = MetricResult("d", 1, None, "cm", Status.LOCKED, provisional=3.0, reasons=("teleport:(1,):100",))
    assert locked.value is None and locked.provisional == 3.0


def test_empty_subject_track_shapes():
    t = SubjectTrack.empty(1, 25, 25.0)
    assert t.n_frames == 25
    assert not t.present.any()
    assert np.isnan(t.cx).all()
    assert (t.source == -1).all()


def test_identity_doubt_blocks_everything_after_it():
    swap = _event(kind=EventKind.CONTACT, subjects=(1, 2), span=Span(300, 320))
    churn = _event(kind=EventKind.ID_CHURN, subjects=(3,), span=Span(500, 501))
    gap = _event(kind=EventKind.GAP, subjects=(1,), span=Span(600, 700))
    events = [swap, churn, gap]
    assert open_blockers(events, [1], Span(0, 300)) == []
    assert open_blockers(events, [2], Span(5000, 6000)) == [swap]
    assert open_blockers(events, [3], Span(9000, 9100)) == [churn]
    assert open_blockers(events, [1], Span(800, 900)) == [swap]
