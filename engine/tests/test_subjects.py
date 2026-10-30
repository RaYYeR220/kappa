import random
import numpy as np
import pytest

from kappa.integrity import event_id
from kappa.model import EventKind, Severity, Span
from kappa.protocol import IntegrityPolicy
from kappa.subjects import Link, assemble, group_tracklets
from synth_detections import FPS, PX_PER_CM, Synth, detection, line, still, wander

POLICY = IntegrityPolicy()  # 150 cm/s, 0.4 s gaps (12 frames), conf 0.5 for 0.5 s (15 frames)


def run(synth: Synth, expected: int = 1, policy: IntegrityPolicy = POLICY, detections=None):
    return assemble(
        synth.detections() if detections is None else detections,
        n_frames=synth.n_frames,
        fps=synth.fps,
        expected_count=expected,
        policy=policy,
        px_per_cm=synth.px_per_cm,
    )


def kinds(events, kind=None, severity=None):
    return [
        e
        for e in events
        if (kind is None or e.kind is kind) and (severity is None or e.severity is severity)
    ]


# -- clean input ---------------------------------------------------------------


def test_clean_single_track_is_reproduced_exactly():
    s = Synth(300, jitter_px=0.7, seed=3).add(1, wander(300, seed=1))
    a = run(s)
    track = a.tracks[1]
    assert track.present.all() and not track.interpolated.any()
    np.testing.assert_array_equal(np.column_stack([track.cx, track.cy]), s.centres(1))
    assert (track.source == 1).all()
    assert a.links == [Link(1, 1, Span(0, 300))]
    assert a.events == []


def test_fields_come_from_detections_and_missing_confidence_is_nan():
    s = Synth(40).add(4, line(40, (100, 100), (2, 1)), zones={"centre": 0.25}).strip_confidence(4)
    t = run(s).tracks[1]
    det = s.detections()[10]
    assert (t.area_px[10], t.major_px[10], t.minor_px[10], t.angle_rad[10]) == (
        det.area_px,
        det.major_px,
        det.minor_px,
        det.angle_rad,
    )
    assert np.isnan(t.confidence).all()
    assert t.fps == FPS and t.n_frames == 40
    np.testing.assert_array_equal(t.zone_fraction["centre"], np.full(40, 0.25))


# -- initial identities ----------------------------------------------------------


def test_subjects_are_numbered_left_to_right_at_first_frame():
    s = (
        Synth(60)
        .add(7, still(60, (600, 300)))
        .add(3, still(60, (200, 500)))
        .add(5, still(60, (400, 100)))
    )
    a = run(s, expected=3)
    assert [int(a.tracks[k].source[0]) for k in (1, 2, 3)] == [3, 5, 7]


def test_initial_identities_are_the_earliest_tracklets():
    path = np.full((60, 2), np.nan)
    path[10:] = (700, 400)
    s = Synth(60).add(1, still(60, (100, 400))).add(2, path).add(3, still(60, (400, 400)))
    a = run(s, expected=2)
    assert {int(a.tracks[k].source[20]) for k in (1, 2)} == {1, 3}


def test_initial_identity_ties_break_on_confidence_then_area():
    s = (
        Synth(60)
        .add(1, still(60, (100, 400)), confidence=0.6)
        .add(2, still(60, (400, 400)), confidence=0.9)
        .add(3, still(60, (700, 400)), confidence=0.9, major_px=90)
    )
    a = run(s, expected=2)
    assert [int(a.tracks[k].source[0]) for k in (1, 2)] == [2, 3]


# -- gaps ----------------------------------------------------------------------


def test_short_gap_is_interpolated_with_one_minor_event():
    s = Synth(120).add(1, line(120, (100, 200), (3, 1))).drop(1, 50, 58)
    a = run(s)
    t = a.tracks[1]
    assert t.present.all()
    assert t.interpolated[50:58].all() and t.interpolated.sum() == 8
    assert (t.source[50:58] == -1).all() and np.isnan(t.confidence[50:58]).all()
    np.testing.assert_allclose(t.cx[50:58], 100 + 3 * np.arange(50, 58))
    np.testing.assert_allclose(t.cy[50:58], 200 + np.arange(50, 58))
    [gap] = a.events
    assert (gap.kind, gap.severity, gap.subjects, gap.span) == (
        EventKind.GAP,
        Severity.MINOR,
        (1,),
        Span(50, 58),
    )


def test_gap_of_exactly_the_tolerance_is_still_interpolated():
    s = Synth(120).add(1, line(120, (100, 200), (3, 1))).drop(1, 50, 62)  # 12 frames = 0.4 s
    [gap] = run(s).events
    assert gap.severity is Severity.MINOR


def test_interpolated_angle_takes_the_short_way_round():
    dets = [
        detection(0, 1, 100, 100, angle_rad=1.5),
        detection(4, 1, 104, 100, angle_rad=-1.5),
    ]
    a = assemble(dets, 5, FPS, 1, POLICY, PX_PER_CM)
    mid = a.tracks[1].angle_rad[2]
    assert abs(abs(mid) - np.pi / 2) < 1e-9  # through vertical, not through horizontal
    assert np.all(np.abs(a.tracks[1].angle_rad) <= np.pi / 2 + 1e-12)


def test_zone_fraction_is_interpolated_across_a_short_gap():
    dets = [
        detection(0, 1, 100, 100, zone_fraction={"centre": 0.0}),
        detection(4, 1, 104, 100, zone_fraction={"centre": 1.0}),
    ]
    t = assemble(dets, 5, FPS, 1, POLICY, PX_PER_CM).tracks[1]
    np.testing.assert_allclose(t.zone_fraction["centre"], [0, 0.25, 0.5, 0.75, 1.0])


def test_long_gap_is_left_absent_and_blocks():
    s = Synth(200).add(1, line(200, (100, 200), (2, 0))).drop(1, 50, 100)
    a = run(s)
    t = a.tracks[1]
    assert not t.present[50:100].any() and np.isnan(t.cx[50:100]).all()
    [gap] = a.events
    assert (gap.kind, gap.severity, gap.span, gap.fix_frame) == (
        EventKind.GAP,
        Severity.BLOCKING,
        Span(50, 100),
        50,
    )
    assert gap.id == "gap:1:50"


def test_absence_before_first_and_after_last_sighting_counts_as_gaps():
    path = np.full((300, 2), np.nan)
    path[20:290] = line(270, (100, 100), (1, 0))
    a = run(Synth(300).add(1, path))
    lead, trail = a.events
    assert (lead.span, lead.severity, lead.fix_frame) == (Span(0, 20), Severity.BLOCKING, 0)
    assert (trail.span, trail.severity) == (Span(290, 300), Severity.MINOR)
    assert not a.tracks[1].present[290:].any()  # never extrapolated


def test_missing_subject_is_one_blocking_gap_over_the_whole_range():
    a = run(Synth(90).add(1, still(90, (100, 100))), expected=2)
    assert not a.tracks[2].present.any()
    [gap] = [e for e in a.events if e.subjects == (2,)]
    assert (gap.kind, gap.severity, gap.span, gap.fix_frame) == (
        EventKind.GAP,
        Severity.BLOCKING,
        Span(0, 90),
        0,
    )


# -- stitching -------------------------------------------------------------------


def test_new_id_after_short_gap_is_stitched_with_minor_churn():
    s = Synth(200).add(1, line(200, (100, 200), (3, 0))).drop(1, 100, 106).reissue(1, 106, 9)
    a = run(s)
    t = a.tracks[1]
    assert t.present.all()
    assert (t.source[:100] == 1).all() and (t.source[106:] == 9).all()
    assert a.links == [Link(1, 1, Span(0, 100)), Link(1, 9, Span(106, 200))]
    churn = kinds(a.events, EventKind.ID_CHURN)
    assert [(e.severity, e.span) for e in churn] == [(Severity.MINOR, Span(100, 107))]
    assert "9" in churn[0].detail
    assert [e.severity for e in kinds(a.events, EventKind.GAP)] == [Severity.MINOR]


def test_new_id_with_no_gap_is_stitched():
    s = Synth(100).add(1, line(100, (100, 200), (3, 0))).reissue(1, 50, 2)
    a = run(s)
    assert a.tracks[1].present.all()
    [churn] = a.events
    assert (churn.kind, churn.severity, churn.span) == (
        EventKind.ID_CHURN,
        Severity.MINOR,
        Span(50, 51),
    )


def test_new_id_after_long_gap_is_stitched_but_the_gap_blocks():
    s = Synth(300).add(1, line(300, (100, 200), (1, 0))).drop(1, 100, 190).reissue(1, 190, 2)
    a = run(s)
    assert (a.tracks[1].source[190:] == 2).all()
    assert [(e.kind, e.severity) for e in a.events] == [
        (EventKind.GAP, Severity.BLOCKING),
        (EventKind.ID_CHURN, Severity.MINOR),
    ]


def test_tracklet_too_far_to_reach_is_not_stitched():
    # 4 frame steps at 150 cm/s and 10 px/cm reach 200 px; the new id appears ~400 px away.
    path = line(200, (100, 200), (1, 0))
    path[103:] += (400, 0)
    s = Synth(200).add(1, path).drop(1, 100, 103).reissue(1, 103, 2)
    a = run(s)
    assert not a.tracks[1].present[100:].any()
    assert [(e.kind, e.severity) for e in a.events] == [
        (EventKind.GAP, Severity.BLOCKING),
        (EventKind.COUNT_MISMATCH, Severity.MINOR),
    ]


def test_several_plausible_continuations_link_the_nearest_and_block():
    path = line(200, (300, 300), (2, 0))
    s = (
        Synth(200)
        .add(1, path)
        .drop(1, 100, 106)
        .reissue(1, 106, 5)
        .add_static(
            6,
            (float(path[106, 0]) + 40, float(path[106, 1])),
            frames=(106, 200),
            major_px=80,
            minor_px=40,
        )
    )
    a = run(s)
    assert (a.tracks[1].source[106:] == 5).all()
    [churn] = kinds(a.events, EventKind.ID_CHURN)
    assert (churn.severity, churn.subjects, churn.fix_frame) == (Severity.BLOCKING, (1,), 106)
    assert "5" in churn.detail and "6" in churn.detail
    [extra] = kinds(a.events, EventKind.COUNT_MISMATCH)
    assert (extra.severity, extra.subjects, extra.span) == (Severity.BLOCKING, (), Span(106, 200))


def test_two_subjects_competing_for_one_tracklet_block_both():
    # Both mice vanish under a shelter; only one comes back out.
    p1, p2 = line(300, (300, 300), (1, 0)), line(300, (300, 360), (1, 0))
    s = Synth(300).add(1, p1).add(2, p2).drop(1, 100, 300).drop(2, 100, 300)
    s.add(3, np.where(np.arange(300)[:, None] >= 130, line(300, (310, 330), (1, 0)), np.nan))
    a = run(s, expected=2)
    [churn] = kinds(a.events, EventKind.ID_CHURN)
    assert (churn.severity, churn.subjects, churn.fix_frame) == (Severity.BLOCKING, (1, 2), 130)
    linked = [k for k in (1, 2) if a.tracks[k].present[130]]
    assert len(linked) == 1 and int(a.tracks[linked[0]].source[130]) == 3


def test_independent_reidentifications_of_two_subjects_stay_minor():
    s = (
        Synth(200)
        .add(1, line(200, (100, 100), (2, 0)))
        .add(2, line(200, (100, 600), (2, 0)))
        .drop(1, 80, 84)
        .reissue(1, 84, 11)
        .drop(2, 82, 86)
        .reissue(2, 86, 12)
    )
    a = run(s, expected=2)
    churn = kinds(a.events, EventKind.ID_CHURN)
    assert [(e.subjects, e.severity) for e in churn] == [
        ((1,), Severity.MINOR),
        ((2,), Severity.MINOR),
    ]
    assert (a.tracks[1].source[84:] == 11).all() and (a.tracks[2].source[86:] == 12).all()


# -- extra objects -------------------------------------------------------------


def test_static_extra_far_from_subjects_is_a_minor_scene_event():
    s = (
        Synth(150)
        .add(1, wander(150, seed=2, centre=(400, 400), reach_px=100))
        .add_static(2, (50, 50))
    )
    a = run(s)
    [extra] = a.events
    assert (extra.kind, extra.severity, extra.subjects, extra.span) == (
        EventKind.COUNT_MISMATCH,
        Severity.MINOR,
        (),
        Span(0, 150),
    )
    assert extra.id == "count_mismatch:scene:0"
    assert (a.tracks[1].source == 1).all()


def test_extra_that_comes_within_a_body_length_of_a_subject_blocks():
    s = Synth(150).add(1, line(150, (100, 400), (4, 0))).add_static(2, (400, 430))
    [extra] = run(s).events
    assert (extra.kind, extra.severity) == (EventKind.COUNT_MISMATCH, Severity.BLOCKING)
    assert extra.fix_frame is not None and 50 <= extra.fix_frame <= 100


def test_brief_extra_detection_is_ignored():
    s = (
        Synth(150).add(1, still(150, (400, 400))).add_static(2, (50, 50), frames=(30, 45))
    )  # 0.5 s, not more
    assert run(s).events == []


def test_frames_with_fewer_subjects_are_gaps_not_count_mismatch():
    s = Synth(150).add(1, still(150, (100, 100))).add(2, still(150, (600, 600))).drop(2, 40, 120)
    a = run(s, expected=2)
    assert [(e.kind, e.subjects) for e in a.events] == [(EventKind.GAP, (2,))]


# -- duplicates ----------------------------------------------------------------


def test_same_frame_duplicate_keeps_the_one_nearest_the_predicted_position():
    s = Synth(60).add(1, line(60, (100, 100), (4, 0))).duplicate(1, 30, (60, 0), confidence=0.99)
    t = run(s).tracks[1]
    assert t.cx[30] == pytest.approx(100 + 4 * 30)
    assert t.confidence[30] == pytest.approx(0.9)


def test_group_tracklets_orders_by_frame():
    dets = [detection(5, 2, 0, 0), detection(1, 2, 0, 0), detection(3, 1, 0, 0)]
    tracklets = group_tracklets(dets)
    assert sorted(tracklets) == [1, 2]
    assert [d.frame for d in tracklets[2].detections] == [1, 5]
    assert tracklets[2].span == Span(1, 6)


# -- determinism and contract ----------------------------------------------------


def _messy_scene() -> Synth:
    return (
        Synth(240, jitter_px=0.5, seed=11)
        .add(1, wander(240, seed=4, centre=(250, 250), reach_px=80))
        .add(2, wander(240, seed=5, centre=(600, 600), reach_px=80))
        .drop(1, 60, 66)
        .reissue(1, 66, 21)
        .drop(2, 100, 160)
        .add_static(9, (300, 600), frames=(10, 200))
        .duplicate(21, 120, (30, -10))
    )


def test_assembly_is_deterministic_regardless_of_input_order():
    s = _messy_scene()
    first = run(s, expected=2)
    shuffled = s.detections()
    random.Random(7).shuffle(shuffled)
    second = run(s, expected=2, detections=shuffled)
    assert first.events == second.events
    assert first.links == second.links
    for k in (1, 2):
        for name in (
            "present",
            "cx",
            "cy",
            "area_px",
            "angle_rad",
            "confidence",
            "source",
            "interpolated",
        ):
            np.testing.assert_array_equal(
                getattr(first.tracks[k], name), getattr(second.tracks[k], name)
            )


def test_event_ids_follow_the_shared_format():
    a = run(_messy_scene(), expected=2)
    assert a.events, "scene should produce events"
    for e in a.events:
        assert e.id == event_id(e.kind, e.subjects, e.span)
        assert e.id == f"{e.kind.value}:{'-'.join(map(str, e.subjects)) or 'scene'}:{e.span.start}"
    assert len({e.id for e in a.events}) == len(a.events)
    assert [e.span.start for e in a.events] == sorted(e.span.start for e in a.events)


def test_rejects_detections_outside_the_analysed_range():
    with pytest.raises(ValueError):
        assemble([detection(10, 1, 0, 0)], 10, FPS, 1, POLICY, PX_PER_CM)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        assemble([], 10, FPS, 0, POLICY, PX_PER_CM)
    with pytest.raises(ValueError):
        assemble([], 10, FPS, 1, POLICY, 0.0)


def test_policy_tolerance_changes_gap_severity():
    s = Synth(120).add(1, line(120, (100, 200), (3, 1))).drop(1, 50, 58)
    strict = IntegrityPolicy(max_interpolated_gap_s=0.1)
    [gap] = run(s, policy=strict).events
    assert gap.severity is Severity.BLOCKING
    assert not run(s, policy=strict).tracks[1].present[50:58].any()
