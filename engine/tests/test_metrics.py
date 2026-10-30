import numpy as np
import pytest
from synth import (
    CENTRE,
    FPS,
    OUTSIDE,
    blocker,
    circle_path,
    dwell,
    jitter,
    line,
    protocol,
    through_centre,
    track,
)

from kappa.metrics import compute_metrics
from kappa.model import BehaviourEvent, EventKind, IntegrityEvent, MetricResult, Severity, Span, Status

GONE = (np.nan, np.nan)
DISTANCE = [{"id": "d", "kind": "distance"}]
ZONE_METRICS = [
    {"id": "t", "kind": "time_in_zone", "zone": "center"},
    {"id": "f", "kind": "fraction_time_in_zone", "zone": "center"},
    {"id": "e", "kind": "zone_entries", "zone": "center"},
    {"id": "l", "kind": "latency_to_zone", "zone": "center"},
]


def run(
    metrics, tracks, *, events=(), behaviour=(), analysed=None, scored=None, **proto
) -> list[MetricResult]:
    tracks = {t.subject: t for t in tracks}
    n = max(t.n_frames for t in tracks.values())
    p = protocol(metrics, **proto)
    if scored is None:
        scored = [b.id for b in p.behaviours]
    return compute_metrics(p, tracks, events, behaviour, analysed or Span(0, n), scored)


def one(results, metric_id=None, *, subject=..., bin=None) -> MetricResult:
    hits = [
        r
        for r in results
        if (metric_id is None or r.metric_id == metric_id)
        and r.bin == bin
        and (subject is ... or r.subject == subject)
    ]
    assert len(hits) == 1, hits
    return hits[0]


def binned(results, metric_id, subject=1) -> list[MetricResult]:
    return sorted(
        (
            r
            for r in results
            if r.metric_id == metric_id and r.subject == subject and r.bin is not None
        ),
        key=lambda r: r.bin,
    )


# --- distance and speed -------------------------------------------------------------


@pytest.mark.parametrize("method", ["none", "median", "savgol"])
def test_distance_and_speed_are_exact_on_a_straight_line(method):
    x, y = line(250, velocity=(4.0, 0.0))
    rs = run(
        DISTANCE + [{"id": "v", "kind": "mean_speed"}], [track(x, y)], smoothing={"method": method}
    )
    d, v = one(rs, "d"), one(rs, "v")
    assert d.status is Status.VERIFIED and d.unit == "cm"
    assert d.value == pytest.approx(4.0 * 249 / FPS)
    assert v.unit == "cm/s" and v.value == pytest.approx(d.value / (250 / FPS))


def test_distance_only_counts_steps_inside_the_analysed_span():
    x, y = line(250, velocity=(4.0, 0.0))
    r = one(run(DISTANCE, [track(x, y)], analysed=Span(50, 150)))
    assert r.value == pytest.approx(4.0 * 99 / FPS)


def test_min_step_suppresses_jitter():
    x, y = jitter(500, amplitude=0.2)
    raw = one(run(DISTANCE, [track(x, y)]))
    damped = one(run(DISTANCE, [track(x, y)], smoothing={"method": "none", "min_step_cm": 1.0}))
    assert raw.value > 20
    assert damped.value == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("method", ["none", "median", "savgol"])
def test_smoothing_never_creates_distance_across_a_gap(method):
    x, y = dwell(((10.0, 10.0), 100), (GONE, 50), ((35.0, 35.0), 100))
    r = one(run(DISTANCE, [track(x, y)], smoothing={"method": method, "window_s": 0.4}))
    assert r.value == pytest.approx(0.0, abs=1e-9)
    assert [e.span for e in r.evidence] == [Span(0, 250), Span(100, 150)]


@pytest.mark.parametrize("method", ["none", "median", "savgol"])
def test_a_gap_drops_only_the_unseen_stretch(method):
    x, y = line(250, velocity=(4.0, 0.0))
    x[100:150] = np.nan
    r = one(run(DISTANCE, [track(x, y)], smoothing={"method": method}))
    assert r.value == pytest.approx(4.0 * (99 + 99) / FPS)


def test_distance_on_interpolated_frames_is_an_uncertainty():
    x, y = line(250, velocity=(4.0, 0.0))
    r = one(run(DISTANCE, [track(x, y, interpolated=range(100, 111))]))
    assert r.uncertainty["interpolated"] == pytest.approx(12 * 4.0 / FPS)
    clean = one(run(DISTANCE, [track(x, y)]))
    assert clean.uncertainty["interpolated"] == 0.0


# --- zones --------------------------------------------------------------------------


def test_time_in_zone_entries_and_latency():
    x, y = dwell((OUTSIDE, 50), (CENTRE, 30), (OUTSIDE, 20), (CENTRE, 5), (OUTSIDE, 45))
    rs = run(ZONE_METRICS, [track(x, y)])
    t, f, e, lat = (one(rs, k) for k in "tfel")
    assert (t.unit, f.unit, e.unit, lat.unit) == ("s", "%", "count", "s")
    assert t.value == pytest.approx(35 / FPS)
    assert f.value == pytest.approx(35 / 150 * 100)
    assert e.value == 2
    assert lat.value == pytest.approx(50 / FPS) and lat.params["censored"] is False
    assert [ev.span for ev in t.evidence] == [Span(50, 80), Span(100, 105)]
    assert [ev.span for ev in e.evidence] == [Span(50, 80), Span(100, 105)]


def test_walking_wall_to_wall_crosses_the_centre_once():
    x, y = through_centre(431)  # 0.1 cm per frame from x = 1 to x = 44
    rs = run(ZONE_METRICS + DISTANCE, [track(x, y)])
    assert one(rs, "d").value == pytest.approx(43.0)
    assert one(rs, "t").value == pytest.approx(225 / FPS)
    assert one(rs, "e").value == 1
    assert one(rs, "l").value == pytest.approx(103 / FPS)


def test_min_dwell_drops_brief_visits_from_entries_but_not_from_time():
    x, y = dwell((OUTSIDE, 50), (CENTRE, 5), (OUTSIDE, 20), (CENTRE, 30), (OUTSIDE, 45))
    rs = run(ZONE_METRICS, [track(x, y)], entry_rule={"min_dwell_s": 0.5})
    assert one(rs, "e").value == 1
    assert one(rs, "l").value == pytest.approx(75 / FPS)
    assert one(rs, "t").value == pytest.approx(35 / FPS)


def test_being_placed_in_the_zone_is_not_an_entry():
    x, y = dwell((CENTRE, 20), (OUTSIDE, 30), (CENTRE, 10), (OUTSIDE, 10))
    rs = run(ZONE_METRICS, [track(x, y)])
    assert one(rs, "e").value == 1
    assert one(rs, "l").value == pytest.approx(50 / FPS)
    assert one(rs, "t").value == pytest.approx(30 / FPS)


def test_never_entering_censors_latency_at_the_span_length():
    x, y = dwell((OUTSIDE, 150))
    rs = run(ZONE_METRICS, [track(x, y)])
    lat = one(rs, "l")
    assert lat.status is Status.VERIFIED
    assert lat.value == pytest.approx(150 / FPS) and lat.params["censored"] is True
    assert one(rs, "e").value == 0 and one(rs, "t").value == 0.0


def test_an_absent_stretch_is_neither_an_exit_nor_an_observed_frame():
    stay = dwell((CENTRE, 20), (GONE, 10), (CENTRE, 20), (OUTSIDE, 10))
    back = dwell((OUTSIDE, 20), (GONE, 10), (CENTRE, 20), (OUTSIDE, 10))
    assert one(run(ZONE_METRICS, [track(*stay)]), "e").value == 0
    rs = run(ZONE_METRICS, [track(*back)])
    assert one(rs, "e").value == 1
    assert one(rs, "f").value == pytest.approx(20 / 50 * 100)


def test_body_fraction_rule_uses_the_mask_overlap_not_the_centroid():
    fraction = np.r_[np.zeros(20), np.full(30, 0.8), np.full(20, 0.5), np.full(30, 0.9)]
    t = track(*dwell((CENTRE, 100)), zone_fraction={"center": fraction})
    rs = run(ZONE_METRICS, [t], entry_rule={"mode": "body_fraction", "body_fraction": 0.7})
    assert one(rs, "t").value == pytest.approx(60 / FPS)
    assert one(rs, "e").value == 2
    assert one(rs, "l").value == pytest.approx(20 / FPS)
    assert one(rs, "t").params["entry_body_fraction"] == 0.7


def test_body_fraction_rule_without_mask_data_is_refused():
    rs = run(ZONE_METRICS, [track(*dwell((CENTRE, 100)))], entry_rule={"mode": "body_fraction"})
    for r in rs:
        assert r.status is Status.REFUSED and r.value is None
        assert "body-fraction" in r.reasons[0]


def test_evidence_is_capped_and_says_how_much_was_left_out():
    x, y = dwell(*[leg for _ in range(60) for leg in ((OUTSIDE, 3), (CENTRE, 2))])
    rs = run(ZONE_METRICS, [track(x, y)])
    t, e = one(rs, "t"), one(rs, "e")
    assert e.value == 60
    assert len(t.evidence) == 50 and len(e.evidence) == 50
    assert "+10 more" in e.evidence[-1].note
    assert all("more" not in ev.note for ev in e.evidence[:-1])


def test_latency_per_bin_restarts_at_each_bin():
    x, y = dwell((OUTSIDE, 60), (CENTRE, 20), (OUTSIDE, 70), (CENTRE, 20), (OUTSIDE, 30))
    rs = run(
        [{"id": "l", "kind": "latency_to_zone", "zone": "center", "bin_s": 2.0}], [track(x, y)]
    )
    assert one(rs, "l").value == pytest.approx(60 / FPS)
    bins = binned(rs, "l")
    assert [r.value for r in bins] == pytest.approx([2.0, 10 / FPS, 2.0, 0.0])
    assert [r.params["censored"] for r in bins] == [True, False, True, False]


# --- immobility ---------------------------------------------------------------------


def test_immobility_bouts_must_last_long_enough():
    moving = np.r_[np.ones(50), np.zeros(75), np.ones(50), np.zeros(12), np.ones(50)].astype(bool)
    x = 5.0 + np.cumsum(np.where(moving, 0.2, 0.0))
    y = np.full_like(x, 22.5)
    rs = run(
        [{"id": "it", "kind": "immobility_time"}, {"id": "ib", "kind": "immobility_bouts"}],
        [track(x, y)],
    )
    it, ib = one(rs, "it"), one(rs, "ib")
    assert ib.value == 1 and ib.unit == "count"
    assert it.value == pytest.approx(3.0, abs=0.1) and it.unit == "s"
    assert len(ib.evidence) == 1 and ib.evidence[0].span.overlaps(Span(50, 125))


# --- bins ---------------------------------------------------------------------------


def _busy_path(n):
    t = np.arange(n) / FPS
    moving = (t % 13.0) < 9.0
    angle = np.cumsum(np.where(moving, 2 * np.pi / 8 / FPS, 0.0))
    return CENTRE[0] + 15 * np.cos(angle), CENTRE[1] + 15 * np.sin(angle)


def test_bins_cover_the_span_and_additive_metrics_sum_to_the_whole():
    n = int(65 * FPS)
    kinds = [
        ("distance", {}),
        ("time_in_zone", {"zone": "center"}),
        ("zone_entries", {"zone": "center"}),
        ("immobility_time", {}),
        ("immobility_bouts", {}),
        ("behaviour_count", {"behaviour": "rearing"}),
        ("behaviour_duration", {"behaviour": "rearing"}),
    ]
    metrics = [{"id": k, "kind": k, "bin_s": 10.0, **extra} for k, extra in kinds]
    behaviour = [
        BehaviourEvent("rearing", 1, s, 0.9, "test")
        for s in (Span(10, 30), Span(240, 260), Span(700, 760), Span(1500, 1625))
    ]
    rs = run(
        metrics,
        [track(*_busy_path(n))],
        behaviour=behaviour,
        smoothing={"method": "median", "min_step_cm": 0.3},
        entry_rule={"min_dwell_s": 0.2},
    )
    for kind, _ in kinds:
        whole, bins = one(rs, kind), binned(rs, kind)
        assert [b.bin for b in bins] == [Span(s, min(s + 250, n)) for s in range(0, n, 250)]
        assert [b.params.get("partial_bin", False) for b in bins] == [False] * 6 + [True]
        assert whole.value > 0, kind
        assert sum(b.value for b in bins) == pytest.approx(whole.value), kind


# --- several subjects ---------------------------------------------------------------

ISD = [{"id": "isd", "kind": "mean_inter_subject_distance"}]


def test_mean_inter_subject_distance():
    a = track(*dwell(((10.0, 10.0), 100)), subject=1)
    b = track(*dwell(((13.0, 10.0), 100)), subject=2)
    c = track(*dwell(((10.0, 14.0), 100)), subject=3)
    r = one(run(ISD, [a, b], subjects=2))
    assert r.subject is None and r.unit == "cm" and r.value == pytest.approx(3.0)
    assert one(run(ISD, [a, b, c], subjects=3)).value == pytest.approx(4.0)


def test_inter_subject_distance_skips_frames_where_anyone_is_missing():
    a = track(*dwell(((10.0, 10.0), 50), ((40.0, 40.0), 50)), subject=1)
    b = track(*dwell(((10.0, 30.0), 50), (GONE, 50)), subject=2)
    r = one(run(ISD, [a, b], subjects=2))
    assert r.value == pytest.approx(20.0)
    assert Span(50, 100) in [e.span for e in r.evidence]


def test_inter_subject_distance_depends_on_every_subject():
    a = track(*dwell(((10.0, 10.0), 100)), subject=1)
    b = track(*dwell(((10.0, 30.0), 100)), subject=2)
    rs = run(ISD + DISTANCE, [a, b], subjects=2, events=[blocker(Span(40, 41), (2,))])
    assert one(rs, "isd").status is Status.LOCKED
    assert one(rs, "d", subject=1).status is Status.VERIFIED
    assert one(rs, "d", subject=2).status is Status.LOCKED


def test_subjects_never_seen_together_are_refused():
    a = track(*dwell(((10.0, 10.0), 50), (GONE, 50)), subject=1)
    b = track(*dwell((GONE, 50), ((10.0, 30.0), 50)), subject=2)
    r = one(run(ISD, [a, b], subjects=2))
    assert r.status is Status.REFUSED and "together" in r.reasons[0]


def test_each_subject_gets_each_metric_and_each_bin():
    a = track(*line(250), subject=1)
    b = track(*line(250, start=(5.0, 10.0)), subject=2)
    rs = run([{"id": "d", "kind": "distance", "bin_s": 4.0}] + ISD, [a, b], subjects=2)
    assert len(rs) == 2 * (1 + 3) + 1
    assert [r.bin for r in binned(rs, "d", 2)] == [Span(0, 100), Span(100, 200), Span(200, 250)]


# --- gating -------------------------------------------------------------------------


def test_a_blocking_event_locks_the_number_and_keeps_it_provisional():
    x, y = line(250, velocity=(4.0, 0.0))
    event = blocker(Span(100, 101))
    r = one(run(DISTANCE, [track(x, y)], events=[event]))
    assert r.status is Status.LOCKED and r.value is None
    assert r.provisional == pytest.approx(4.0 * 249 / FPS)
    assert r.reasons == (event.id,)
    assert r.evidence and r.params


@pytest.mark.parametrize(
    "event",
    [
        blocker(Span(100, 101), resolved_by="repair-1"),
        blocker(Span(100, 101), severity=Severity.MINOR),
        blocker(Span(100, 101), (2,)),
        blocker(Span(200, 260)),
    ],
    ids=["resolved", "minor", "other-subject", "outside-span"],
)
def test_events_that_do_not_block_leave_the_number_verified(event):
    x, y = line(250, velocity=(4.0, 0.0))
    r = one(run(DISTANCE, [track(x, y)], events=[event], analysed=Span(0, 200)))
    assert r.status is Status.VERIFIED and r.value == pytest.approx(4.0 * 199 / FPS)


def test_scene_level_events_lock_every_subject():
    x, y = line(250)
    event = blocker(Span(10, 20), (), kind=EventKind.COUNT_MISMATCH)
    assert one(run(DISTANCE, [track(x, y)], events=[event])).status is Status.LOCKED


def test_only_the_bins_an_event_touches_are_locked():
    x, y = line(250, velocity=(4.0, 0.0))
    events = [blocker(Span(120, 125)), blocker(Span(195, 205), kind=EventKind.GAP)]
    rs = run([{"id": "d", "kind": "distance", "bin_s": 2.0}], [track(x, y)], events=events)
    assert one(rs, "d").status is Status.LOCKED
    assert one(rs, "d").reasons == tuple(e.id for e in events)
    statuses = [r.status for r in binned(rs, "d")]
    assert statuses == [
        Status.VERIFIED,
        Status.VERIFIED,
        Status.LOCKED,
        Status.LOCKED,
        Status.LOCKED,
    ]
    assert binned(rs, "d")[2].reasons == (events[0].id,)


def test_a_subject_never_observed_is_refused():
    seen = track(*line(250), subject=1)
    unseen = track(*dwell((GONE, 250)), subject=2)
    rs = run(DISTANCE + ZONE_METRICS, [seen, unseen], subjects=2)
    for r in rs:
        if r.subject == 2:
            assert r.status is Status.REFUSED and "never observed" in r.reasons[0]
        else:
            assert r.status is Status.VERIFIED


def test_a_bin_where_the_subject_was_absent_is_refused_alone():
    x, y = line(250)
    x[:50] = np.nan
    rs = run([{"id": "d", "kind": "distance", "bin_s": 2.0}], [track(x, y)])
    assert one(rs, "d").status is Status.VERIFIED
    assert [r.status for r in binned(rs, "d")] == [Status.REFUSED] + [Status.VERIFIED] * 4


def test_an_unprojected_track_cannot_be_measured_in_cm():
    t = track(*line(250))
    t.x_cm = t.y_cm = None
    rs = run(
        DISTANCE
        + ZONE_METRICS[:1]
        + [{"id": "b", "kind": "behaviour_count", "behaviour": "rearing"}],
        [t],
    )
    assert one(rs, "d").status is Status.REFUSED and "arena" in one(rs, "d").reasons[0]
    assert one(rs, "t").status is Status.REFUSED
    assert one(rs, "b").status is Status.VERIFIED


def test_no_tracks_refuses_every_metric():
    rs = compute_metrics(protocol(DISTANCE + ISD, subjects=2), {}, [], [], Span(0, 100))
    assert [(r.metric_id, r.subject, r.status) for r in rs] == [
        ("d", None, Status.REFUSED),
        ("isd", None, Status.REFUSED),
    ]


def test_a_span_past_the_end_of_the_recording_counts_as_unobserved():
    rs = run(
        DISTANCE + ZONE_METRICS, [track(*dwell((OUTSIDE, 50), (CENTRE, 50)))], analysed=Span(0, 200)
    )
    assert one(rs, "f").value == pytest.approx(50.0)
    assert one(rs, "d").status is Status.VERIFIED


# --- behaviour ----------------------------------------------------------------------


def test_behaviour_bouts_are_clipped_to_the_span():
    events = [
        BehaviourEvent("rearing", 1, Span(40, 60), 0.9, "test"),
        BehaviourEvent("rearing", 1, Span(100, 125), 0.9, "test"),
        BehaviourEvent("rearing", 1, Span(110, 130), 0.9, "test"),
        BehaviourEvent("rearing", 1, Span(240, 260), 0.9, "test"),
        BehaviourEvent("rearing", 1, Span(10, 20), 0.9, "test"),
        BehaviourEvent("rearing", 2, Span(150, 160), 0.9, "test"),
        BehaviourEvent("grooming", 1, Span(150, 160), 0.9, "test"),
    ]
    metrics = [
        {"id": "n", "kind": "behaviour_count", "behaviour": "rearing"},
        {"id": "dur", "kind": "behaviour_duration", "behaviour": "rearing"},
    ]
    rs = run(metrics, [track(*dwell((OUTSIDE, 300)))], behaviour=events, analysed=Span(50, 250))
    n, dur = one(rs, "n"), one(rs, "dur")
    assert n.value == 3 and n.unit == "count"
    assert dur.value == pytest.approx((10 + 30 + 10) / FPS) and dur.unit == "s"
    assert [e.span for e in dur.evidence] == [Span(50, 60), Span(100, 130), Span(240, 250)]


# --- params -------------------------------------------------------------------------


def test_params_record_the_method():
    metrics = (
        ZONE_METRICS
        + DISTANCE
        + [{"id": "it", "kind": "immobility_time", "immobile_below_cm_s": 1.5}]
    )
    rs = run(
        metrics,
        [track(*circle_path(250))],
        entry_rule={"min_dwell_s": 0.5},
        smoothing={"method": "median", "window_s": 0.2, "min_step_cm": 0.5},
    )
    t, e, d, it = one(rs, "t"), one(rs, "e"), one(rs, "d"), one(rs, "it")
    assert t.params["zone"] == "center" and t.params["zone_kind"] == "center_fraction"
    assert t.params["zone_area_fraction"] == 0.25 and t.params["entry_mode"] == "centroid"
    assert t.params["fps"] == FPS and t.params["duration_s"] == pytest.approx(10.0)
    assert e.params["min_dwell_s"] == 0.5
    assert d.params["smoothing"] == "median" and d.params["smoothing_window_frames"] == 5
    assert d.params["min_step_cm"] == 0.5
    assert it.params["immobile_below_cm_s"] == 1.5 and it.params["immobile_min_s"] == 1.0
    assert "min_step_cm" not in it.params


# --- behaviour scoring status -------------------------------------------------------


def test_unscored_behaviour_is_refused_not_zero():
    metrics = [{"id": "n", "kind": "behaviour_count", "behaviour": "rearing"}]
    r = one(run(metrics, [track(*dwell((OUTSIDE, 300)))], scored=[]), "n")
    assert r.status is Status.REFUSED
    assert "not been scored" in r.reasons[0]
    scored = one(run(metrics, [track(*dwell((OUTSIDE, 300)))]), "n")
    assert scored.status is Status.VERIFIED and scored.value == 0


def test_incomplete_scoring_locks_only_the_bins_it_touches():
    metrics = [{"id": "n", "kind": "behaviour_count", "behaviour": "rearing", "bin_s": 100 / FPS}]
    gap = IntegrityEvent(
        "scoring_incomplete:1:120", EventKind.SCORING_INCOMPLETE, Severity.BLOCKING, (1,), Span(120, 140), "2 verdicts failed"
    )
    rs = run(metrics, [track(*dwell((OUTSIDE, 300)))], events=[gap])
    assert one(rs, "n").status is Status.LOCKED
    statuses = [r.status for r in binned(rs, "n")]
    assert statuses == [Status.VERIFIED, Status.LOCKED, Status.VERIFIED]


def test_zero_smoothing_window_means_no_smoothing():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    y = np.zeros_like(x)
    r = one(run(DISTANCE, [track(x, y)], smoothing={"method": "median", "window_s": 0.0}))
    assert r.value == pytest.approx(5.0)


def test_identity_doubt_does_not_lock_identity_free_group_numbers():
    import numpy as np
    from synth import track as _t

    a = _t(np.linspace(10, 20, 100), np.full(100, 10.0), subject=1)
    b = _t(np.linspace(30, 20, 100), np.full(100, 30.0), subject=2)
    swap = IntegrityEvent("contact:1-2:40", EventKind.CONTACT, Severity.BLOCKING, (1, 2), Span(40, 60), "ambiguous")
    rs = run([{"id": "isd", "kind": "mean_inter_subject_distance"}, {"id": "d", "kind": "distance"}], [a, b], events=[swap], subjects=2)
    assert one(rs, "isd").status is Status.VERIFIED
    assert all(r.status is Status.LOCKED for r in rs if r.metric_id == "d")
    gap = IntegrityEvent("gap:1:50", EventKind.GAP, Severity.BLOCKING, (1,), Span(50, 70), "lost")
    rs = run([{"id": "isd", "kind": "mean_inter_subject_distance"}], [a, b], events=[gap], subjects=2)
    assert one(rs, "isd").status is Status.LOCKED
