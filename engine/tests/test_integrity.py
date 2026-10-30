import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kappa.integrity import (
    RESOLVED_KEEP,
    RESOLVED_SWAP,
    all_events,
    check,
    contact_episodes,
    event_id,
    resolve,
    resolve_contacts,
    scan,
    summary,
)
from kappa.model import EventKind, Severity, Span, open_blockers
from kappa.protocol import IntegrityPolicy
from kappa.subjects import Link, assemble
from synth_detections import (
    PX_PER_CM,
    Synth,
    closest_approach,
    crossing,
    line,
    still,
    touch_and_turn_back,
    wander,
    waypoints,
)

POLICY = IntegrityPolicy()
POST = 9  # 0.3 s at 30 fps


def assembled(
    synth: Synth,
    expected: int = 1,
    policy: IntegrityPolicy = POLICY,
    px_per_cm: float | None = None,
):
    return assemble(
        synth.detections(),
        n_frames=synth.n_frames,
        fps=synth.fps,
        expected_count=expected,
        policy=policy,
        px_per_cm=synth.px_per_cm if px_per_cm is None else px_per_cm,
    )


def of(events, kind):
    return [e for e in events if e.kind is kind]


def centres(track):
    return np.column_stack([track.cx, track.cy])


# -- teleport --------------------------------------------------------------------


def test_one_frame_jump_is_a_blocking_teleport_out_and_back():
    s = Synth(120).add(1, line(120, (100, 300), (3, 0))).teleport(1, 60, (300, 0))
    events = scan(assembled(s).tracks, POLICY, PX_PER_CM)
    jumps = of(events, EventKind.TELEPORT)
    assert [(e.span, e.fix_frame, e.severity) for e in jumps] == [
        (Span(59, 61), 60, Severity.BLOCKING),
        (Span(60, 62), 61, Severity.BLOCKING),
    ]
    assert jumps[0].id == "teleport:1:59"


def test_teleport_threshold_is_in_cm_per_second():
    # A 40 px one-frame jump at 30 fps: 120 cm/s at 10 px/cm, 240 cm/s at 5 px/cm.
    s = Synth(60).add(1, still(60, (300, 300))).teleport(1, 30, (40, 0))
    assert of(scan(assembled(s).tracks, POLICY, 10.0), EventKind.TELEPORT) == []
    assert len(of(scan(assembled(s, px_per_cm=5.0).tracks, POLICY, 5.0), EventKind.TELEPORT)) == 2


def test_no_teleport_across_an_interpolated_gap():
    s = Synth(120).add(1, line(120, (100, 300), (4, 0))).drop(1, 40, 50)
    assert of(scan(assembled(s).tracks, POLICY, PX_PER_CM), EventKind.TELEPORT) == []


# -- low confidence --------------------------------------------------------------


def test_low_confidence_runs_block_when_long_and_are_minor_when_short():
    s = (
        Synth(200)
        .add(1, still(200, (300, 300)))
        .low_confidence(1, 20, 35)
        .low_confidence(1, 100, 105)
    )
    lows = of(scan(assembled(s).tracks, POLICY, PX_PER_CM), EventKind.LOW_CONFIDENCE)
    assert [(e.span, e.severity, e.fix_frame) for e in lows] == [
        (Span(20, 35), Severity.BLOCKING, 20),  # exactly 0.5 s
        (Span(100, 105), Severity.MINOR, None),
    ]


def test_missing_confidence_is_not_low():
    s = Synth(100).add(1, still(100, (300, 300))).strip_confidence(1)
    assert scan(assembled(s).tracks, POLICY, PX_PER_CM) == []


def test_low_confidence_run_is_bridged_across_dropouts():
    s = (
        Synth(100)
        .add(1, line(100, (100, 300), (2, 0)))
        .low_confidence(1, 20, 28)
        .drop(1, 28, 32)
        .low_confidence(1, 32, 40)
    )
    [low] = of(scan(assembled(s).tracks, POLICY, PX_PER_CM), EventKind.LOW_CONFIDENCE)
    assert (low.span, low.severity) == (Span(20, 40), Severity.BLOCKING)


# -- contacts --------------------------------------------------------------------


def _crossing_scene(n=120, swap=False, seed=1):
    a, b = crossing(n)
    s = Synth(n, jitter_px=0.5, seed=seed).add(1, a).add(2, b)
    if swap:
        s.swap_ids(1, 2, closest_approach(a, b))
    return s


def test_clean_crossing_keeps_identities_by_motion_continuity():
    asm = assembled(_crossing_scene(), expected=2)
    fixed, events = resolve_contacts(asm.tracks, POLICY, PX_PER_CM)
    [contact] = events
    assert (contact.kind, contact.severity, contact.subjects, contact.resolved_by) == (
        EventKind.CONTACT,
        Severity.MINOR,
        (1, 2),
        RESOLVED_KEEP,
    )
    for k in (1, 2):
        np.testing.assert_array_equal(fixed[k].cx, asm.tracks[k].cx)


def test_contact_span_is_the_episode_plus_the_post_window():
    asm = assembled(_crossing_scene(), expected=2)
    [episode] = contact_episodes(asm.tracks[1], asm.tracks[2], POLICY, merge_frames=POST)
    assert 45 < episode.start < 60 and 65 < episode.end < 80
    [contact] = of(scan(asm.tracks, POLICY, PX_PER_CM), EventKind.CONTACT)
    assert contact.span == Span(episode.start, episode.end + POST)
    assert contact.id == f"contact:1-2:{episode.start}"


def test_segmenter_id_swap_at_a_crossing_is_swapped_back():
    clean = _crossing_scene()
    asm = assembled(_crossing_scene(swap=True), expected=2)
    assert not np.array_equal(centres(asm.tracks[1])[100:], clean.centres(1)[100:])
    fixed, [contact] = resolve_contacts(asm.tracks, POLICY, PX_PER_CM)
    assert (contact.severity, contact.resolved_by) == (Severity.MINOR, RESOLVED_SWAP)
    np.testing.assert_array_equal(centres(fixed[1]), clean.centres(1))
    np.testing.assert_array_equal(centres(fixed[2]), clean.centres(2))
    # The correction must not itself look like a jump.
    assert of(check(asm, POLICY, PX_PER_CM).events, EventKind.TELEPORT) == []


def test_swap_correction_leaves_the_input_tracks_alone():
    asm = assembled(_crossing_scene(swap=True), expected=2)
    before = centres(asm.tracks[1]).copy()
    resolve_contacts(asm.tracks, POLICY, PX_PER_CM)
    scan(asm.tracks, POLICY, PX_PER_CM)
    check(asm, POLICY, PX_PER_CM)
    np.testing.assert_array_equal(centres(asm.tracks[1]), before)


@pytest.mark.parametrize("segmenter_swapped", [False, True])
def test_touch_and_turn_back_stays_ambiguous(segmenter_swapped):
    a, b = touch_and_turn_back(150)
    s = Synth(150, jitter_px=0.5, seed=2).add(1, a).add(2, b)
    if segmenter_swapped:
        s.swap_ids(1, 2, 75)
    asm = assembled(s, expected=2)
    fixed, [contact] = resolve_contacts(asm.tracks, POLICY, PX_PER_CM)
    [episode] = contact_episodes(asm.tracks[1], asm.tracks[2], POLICY, merge_frames=POST)
    assert (contact.severity, contact.resolved_by, contact.fix_frame) == (
        Severity.BLOCKING,
        None,
        episode.end,
    )
    assert "cannot tell" in contact.detail
    np.testing.assert_array_equal(fixed[1].cx, asm.tracks[1].cx)  # a guess is never applied


def _head_on(segmenter_swap_at=None):
    # Nose to nose on one line: stop 40 px apart, pause half a second, back off.
    a = waypoints(135, [(0, 100, 400), (60, 380, 400), (75, 380, 400), (135, 100, 400)])
    b = waypoints(135, [(0, 700, 400), (60, 420, 400), (75, 420, 400), (135, 700, 400)])
    s = Synth(135, jitter_px=0.5, seed=5).add(1, a).add(2, b)
    if segmenter_swap_at is not None:
        s.swap_ids(1, 2, segmenter_swap_at)
    return assembled(s, expected=2)


def test_turn_back_that_extrapolates_like_a_pass_through_is_not_swapped():
    # Carried forward, each animal's approach lands on the other: extrapolation alone
    # would "correct" ids that were never wrong. They never touched closer than 40 px,
    # so the segmenter could not have crossed the ids without a jump.
    asm = _head_on()
    fixed, [contact] = resolve_contacts(asm.tracks, POLICY, PX_PER_CM)
    assert (contact.severity, contact.resolved_by) == (Severity.BLOCKING, None)
    assert "cannot have crossed" in contact.detail
    np.testing.assert_array_equal(fixed[1].cx, asm.tracks[1].cx)


def test_id_jump_inside_a_contact_contradicts_a_confident_keep():
    # The segmenter swaps ids mid-pause: a 40 px jump (120 cm/s, under the teleport
    # limit) after which the reported tracks look like a clean pass-through.
    asm = _head_on(segmenter_swap_at=68)
    events = check(asm, POLICY, PX_PER_CM).events
    assert of(events, EventKind.TELEPORT) == []
    [contact] = of(events, EventKind.CONTACT)
    assert (contact.severity, contact.resolved_by) == (Severity.BLOCKING, None)
    assert "frame 68" in contact.detail


def test_contact_lasting_to_the_end_is_minor():
    a, b = crossing(70, cross_frame=65)
    [contact] = scan(assembled(Synth(70).add(1, a).add(2, b), expected=2).tracks, POLICY, PX_PER_CM)
    assert (contact.kind, contact.severity, contact.span.end) == (
        EventKind.CONTACT,
        Severity.MINOR,
        70,
    )


def test_contact_from_the_first_frame_blocks():
    a, b = crossing(60, cross_frame=3)
    [contact] = scan(assembled(Synth(60).add(1, a).add(2, b), expected=2).tracks, POLICY, PX_PER_CM)
    assert (contact.kind, contact.severity, contact.span.start) == (
        EventKind.CONTACT,
        Severity.BLOCKING,
        0,
    )


def test_third_subject_far_away_is_untouched():
    s = _crossing_scene(swap=True).add(3, wander(120, seed=9, centre=(900, 900), reach_px=50))
    asm = assembled(s, expected=3)
    fixed, events = resolve_contacts(asm.tracks, POLICY, PX_PER_CM)
    assert [e.subjects for e in events] == [(1, 2)]
    np.testing.assert_array_equal(fixed[3].cx, asm.tracks[3].cx)


def test_two_crossings_with_swaps_are_both_corrected():
    # A walks right along y=400 and meets B twice: B comes down through A's line at
    # x=400 (frame ~64), heads right below it, then comes back up through it at x=1000.
    n = 240
    a = line(n, (100, 400), (5, 0))
    b = waypoints(n, [(0, 400, 80), (94, 400, 550), (154, 1000, 550), (240, 1000, 120)])
    clean = Synth(n, jitter_px=0.5, seed=4).add(1, a).add(2, b)
    s = Synth(n, jitter_px=0.5, seed=4).add(1, a).add(2, b)
    s.swap_ids(1, 2, closest_approach(a[:120], b[:120]))
    s.swap_ids(1, 2, 120 + closest_approach(a[120:], b[120:]))
    checked = check(assembled(s, expected=2), POLICY, PX_PER_CM)
    contacts = of(checked.events, EventKind.CONTACT)
    assert [e.resolved_by for e in contacts] == [RESOLVED_SWAP, RESOLVED_SWAP]
    np.testing.assert_array_equal(centres(checked.tracks[1]), clean.centres(1))
    np.testing.assert_array_equal(centres(checked.tracks[2]), clean.centres(2))


# -- putting it together ---------------------------------------------------------


def test_check_relabels_assembly_events_and_links_after_a_swap():
    a, b = crossing(120)
    k = closest_approach(a, b)
    s = _crossing_scene(swap=True).drop(1, 90, 95)  # after the swap, track 1 is animal B
    asm = assembled(s, expected=2)
    assert [e.id for e in of(asm.events, EventKind.GAP)] == ["gap:1:90"]
    checked = check(asm, POLICY, PX_PER_CM)
    [gap] = of(checked.events, EventKind.GAP)
    assert (gap.id, gap.subjects, gap.span) == ("gap:2:90", (2,), Span(90, 95))
    assert checked.tracks[2].interpolated[90:95].all()
    assert checked.links == [
        Link(1, 1, Span(0, k)),
        Link(1, 2, Span(k, 120)),
        Link(2, 2, Span(0, k)),
        Link(2, 1, Span(k, 120)),
    ]


def _messy():
    a, b = crossing(150, cross_frame=70)
    s = (
        Synth(150, jitter_px=0.5, seed=8)
        .add(1, a)
        .add(2, b)
        .swap_ids(1, 2, closest_approach(a, b))
        .teleport(1, 20, (200, 0))
        .low_confidence(2, 100, 130)
        .drop(2, 5, 9)
        .add_static(7, (60, 700))
    )
    return assembled(s, expected=2)


def test_all_events_merges_sorts_and_deduplicates():
    asm = _messy()
    events = all_events(asm, POLICY, PX_PER_CM)
    assert {e.kind for e in events} >= {
        EventKind.GAP,
        EventKind.COUNT_MISMATCH,
        EventKind.CONTACT,
        EventKind.TELEPORT,
        EventKind.LOW_CONFIDENCE,
    }
    assert [e.span.start for e in events] == sorted(e.span.start for e in events)
    assert len({e.id for e in events}) == len(events)
    for e in events:
        assert e.id == event_id(e.kind, e.subjects, e.span)
    again = check(check(asm, POLICY, PX_PER_CM), POLICY, PX_PER_CM)
    assert [e.id for e in again.events] == [e.id for e in events]


def test_integrity_is_deterministic():
    first, second = all_events(_messy(), POLICY, PX_PER_CM), all_events(_messy(), POLICY, PX_PER_CM)
    assert first == second


def test_resolve_marks_one_event_and_unlocks_it():
    events = all_events(_messy(), POLICY, PX_PER_CM)
    blocker = next(e for e in events if e.is_open_blocker)
    fixed = resolve(events, blocker.id, "repair-1")
    [after] = [e for e in fixed if e.id == blocker.id]
    assert after.resolved_by == "repair-1" and not after.is_open_blocker
    assert [e for e in fixed if e.id != blocker.id] == [e for e in events if e.id != blocker.id]
    assert blocker.resolved_by is None
    whole = Span(0, 150)
    assert len(open_blockers(fixed, None, whole)) == len(open_blockers(events, None, whole)) - 1
    with pytest.raises(KeyError):
        resolve(events, "gap:9:0", "repair-1")
    with pytest.raises(ValueError):
        resolve(events, blocker.id, "")


def test_summary_counts_kinds_severities_and_open_blockers():
    events = all_events(_messy(), POLICY, PX_PER_CM)
    report = summary(events)
    assert report["total"] == len(events)
    assert sum(sum(v.values()) for v in report["by_kind"].values()) == len(events)
    assert report["by_severity"]["blocking"] == sum(e.severity is Severity.BLOCKING for e in events)
    assert report["open_blocker_ids"] == [e.id for e in events if e.is_open_blocker]
    assert report["open_blockers"] == len(report["open_blocker_ids"])
    resolved = resolve(events, report["open_blocker_ids"][0], "repair-1")
    assert summary(resolved)["open_blockers"] == report["open_blockers"] - 1
    assert summary([]) == {
        "total": 0,
        "by_kind": {},
        "by_severity": {"minor": 0, "blocking": 0},
        "resolved": 0,
        "open_blockers": 0,
        "open_blocker_ids": [],
    }


# -- property --------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(20, 400),
    seed=st.integers(0, 2**32 - 1),
    fps=st.sampled_from([15.0, 25.0, 30.0, 60.0]),
    px_per_cm=st.floats(2.0, 30.0),
    speed_cm_s=st.floats(0.5, 140.0),
    confidence=st.one_of(st.none(), st.floats(0.5, 1.0)),
    order=st.randoms(use_true_random=False),
)
def test_clean_single_subject_never_blocks_and_is_reproduced_exactly(
    n, seed, fps, px_per_cm, speed_cm_s, confidence, order
):
    path = wander(n, seed, speed_cm_s=speed_cm_s, fps=fps, px_per_cm=px_per_cm)
    s = Synth(n, fps=fps, px_per_cm=px_per_cm).add(1, path, confidence=confidence)
    detections = s.detections()
    order.shuffle(detections)
    asm = assemble(detections, n, fps, 1, POLICY, px_per_cm)
    checked = check(asm, POLICY, px_per_cm)
    assert [e for e in checked.events if e.severity is Severity.BLOCKING] == []
    assert checked.events == []
    track = checked.tracks[1]
    assert track.present.all() and not track.interpolated.any()
    np.testing.assert_array_equal(track.cx, path[:, 0])
    np.testing.assert_array_equal(track.cy, path[:, 1])
