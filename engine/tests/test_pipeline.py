"""End to end on synthetic segmenter output: raw text in, gated numbers out."""

import numpy as np
import pytest
from sam_synth import ellipse_mask, line, record
from synth_detections import touch_and_turn_back

from kappa.model import EventKind, Severity, Status
from kappa.pipeline import NeedsCalibration, Repair, analyse
from kappa.protocol import AssayProtocol
from kappa.repair import Anchor, prompt_objects
from kappa.video import VideoInfo

FPS, N, SIZE = 30.0, 150, 800
VIDEO = VideoInfo(width=SIZE, height=SIZE, fps=FPS, n_frames=N, duration_s=N / FPS, codec="h264")
MASK = ellipse_mask(20, 40)


def protocol(count=2, corners=((0, 0), (800, 0), (800, 800), (0, 800))):
    return AssayProtocol.model_validate(
        {
            "title": "t",
            "question": "q",
            "subject": {"concept": "mouse", "expected_count": count},
            "arena": {"shape": "rect", "width_cm": 80, "height_cm": 80, "corners_px": corners},
            "zones": [{"id": "center", "kind": "center_fraction", "fraction": 0.25}],
            "metrics": [
                {"id": "distance", "kind": "distance"},
                {"id": "center_time", "kind": "time_in_zone", "zone": "center"},
            ],
            "smoothing": {"method": "none"},
            "integrity": {"contact_distance": 1.0},
        }
    )


def render(paths: dict[int, np.ndarray], frames: range) -> str:
    out = []
    h, w = MASK.shape
    for f in frames:
        recs = [
            record(obj, int(round(p[f, 0] - w / 2)), int(round(p[f, 1] - h / 2)), MASK, SIZE, SIZE, 0.95)
            for obj, p in paths.items()
            if not np.isnan(p[f, 0])
        ]
        if recs:
            out.append(line(f, recs))
    return "\n".join(out) + "\n"


def test_single_subject_straight_walk_is_verified():
    path = np.column_stack([np.linspace(100, 700, N), np.full(N, 400.0)])
    a = analyse(protocol(count=1), VIDEO, render({0: path}, range(N)), workers=1)
    distance = next(r for r in a.results if r.metric_id == "distance" and r.bin is None)
    assert distance.status is Status.VERIFIED
    assert distance.value == pytest.approx(60.0, rel=0.01)
    centre = next(r for r in a.results if r.metric_id == "center_time")
    assert centre.status is Status.VERIFIED and centre.value > 0
    assert not [e for e in a.events if e.is_open_blocker]


def test_ambiguous_contact_locks_then_a_person_settles_it():
    a_path, b_path = touch_and_turn_back(N)
    raw = render({7: a_path, 9: b_path}, range(N))
    first = analyse(protocol(), VIDEO, raw, workers=1)
    blockers = [e for e in first.events if e.is_open_blocker]
    assert any(e.kind is EventKind.CONTACT and e.severity is Severity.BLOCKING for e in blockers)
    locked = [r for r in first.results if r.status is Status.LOCKED]
    assert {r.subject for r in locked} == {1, 2}
    contact = next(e for e in blockers if e.kind is EventKind.CONTACT)

    fix = contact.fix_frame
    anchors = [
        Anchor(1, fix, point=(int(a_path[fix, 0]), int(a_path[fix, 1]))),
        Anchor(2, fix, point=(int(b_path[fix, 0]), int(b_path[fix, 1]))),
    ]
    assert [o.id for o in prompt_objects(anchors)] == [1, 2]
    retrack = render({1: a_path, 2: b_path}, range(fix, N))
    second = analyse(protocol(), VIDEO, raw, repairs=[Repair(fix, retrack, "a person")], workers=1)
    settled = next(e for e in second.events if e.id == contact.id)
    assert settled.resolved_by == "a person"
    distances = [r for r in second.results if r.metric_id == "distance" and r.bin is None]
    assert all(r.status is Status.VERIFIED for r in distances), [r.reasons for r in distances]


def test_unknown_corners_ask_for_calibration():
    with pytest.raises(NeedsCalibration):
        analyse(protocol(corners=None), VIDEO, "", workers=1)


def test_analysis_is_deterministic():
    a_path, b_path = touch_and_turn_back(N)
    raw = render({7: a_path, 9: b_path}, range(N))
    one, two = analyse(protocol(), VIDEO, raw, workers=1), analyse(protocol(), VIDEO, raw, workers=1)
    assert [e.id for e in one.events] == [e.id for e in two.events]
    assert [(r.metric_id, r.subject, r.status, r.value, r.provisional) for r in one.results] == [
        (r.metric_id, r.subject, r.status, r.value, r.provisional) for r in two.results
    ]
