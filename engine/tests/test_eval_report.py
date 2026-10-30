import json
import math

import numpy as np
import pytest

from kappa.eval.report import (
    AgreementReport,
    aggregate,
    bouts_from_events,
    compare_to_raters,
    pairwise_agreement,
)
from kappa.eval.stats import icc, pearson_r
from kappa.model import BehaviourEvent, Span

FPS = 25.0
DURATION_S = 600.0
N_FRAMES = int(DURATION_S * FPS)


def random_bouts(rng, rate_per_s=0.05, mean_len_s=1.5, duration_s=DURATION_S):
    """Non-overlapping bouts with exponential gaps and lengths."""
    bouts, t = [], 0.0
    while True:
        t += rng.exponential(1 / rate_per_s)
        length = max(0.2, rng.exponential(mean_len_s))
        if t + length >= duration_s:
            return bouts
        bouts.append((t, t + length))
        t += length


def noisy_copy(rng, bouts, jitter_s=0.15, drop=0.05):
    """A second rater: same bouts, slightly different boundaries, a few missed."""
    out = []
    for onset, offset in bouts:
        if rng.random() < drop:
            continue
        a, b = onset + rng.normal(0, jitter_s), offset + rng.normal(0, jitter_s)
        if b > a:
            out.append((max(a, 0.0), b))
    return out


def rater_panel(seed):
    rng = np.random.default_rng(seed)
    truth = random_bouts(rng)
    return truth, {name: noisy_copy(rng, truth) for name in ("ann", "bob", "cat")}


def test_instrument_identical_to_a_rater():
    _, raters = rater_panel(0)
    rep = compare_to_raters(raters["ann"], raters, fps=FPS, n_frames=N_FRAMES)
    by_rater = {p.b: p for p in rep.vs_raters}
    assert by_rater["ann"].kappa == 1.0
    assert by_rater["ann"].bout_f1 == 1.0
    assert rep.instrument_count == rep.rater_counts["ann"] == len(raters["ann"])
    assert rep.within_human_range is True
    assert [(p.a, p.b) for p in rep.between_raters] == [
        ("ann", "bob"),
        ("ann", "cat"),
        ("bob", "cat"),
    ]
    lo, hi = rep.human_range
    assert 0.6 < lo <= hi < 1.0
    assert rep.instrument_mean_kappa == pytest.approx(np.mean([p.kappa for p in rep.vs_raters]))


def test_counts_and_durations_are_measured_inside_the_window():
    raters = {"ann": [(1.0, 2.0), (3.0, 3.5), (100.0, 101.0)], "bob": [(1.0, 2.0)]}
    rep = compare_to_raters(
        [(0.0, 50.0)], raters, fps=10.0, n_frames=100, offset_s=0.5
    )  # window 0.5..10.5 s
    assert rep.rater_counts == {"ann": 2, "bob": 1}
    assert rep.rater_durations_s["ann"] == pytest.approx(1.5)
    assert rep.instrument_duration_s == pytest.approx(10.0)


# ---------------------------------------------------------------- negative controls


def test_random_bouts_score_kappa_near_zero():
    kappas = []
    for seed in range(5):
        _, raters = rater_panel(seed)
        instrument = random_bouts(np.random.default_rng(1000 + seed))
        rep = compare_to_raters(instrument, raters, fps=FPS, n_frames=N_FRAMES)
        kappas.append(rep.instrument_mean_kappa)
        assert rep.within_human_range is False
    assert abs(np.mean(kappas)) < 0.05
    assert max(abs(k) for k in kappas) < 0.1


def test_all_false_prediction_scores_zero():
    _, raters = rater_panel(1)
    rep = compare_to_raters([], raters, fps=FPS, n_frames=N_FRAMES)
    assert all(p.kappa == 0.0 for p in rep.vs_raters)
    assert all(p.bout_f1 == 0.0 for p in rep.vs_raters)
    assert rep.instrument_count == 0 and rep.instrument_duration_s == 0.0
    assert rep.within_human_range is False


def test_shuffling_rater_identities_keeps_the_human_range():
    _, raters = rater_panel(2)
    names = list(raters)
    shuffled = dict(zip(["zed", "amy", "kim"], [raters[n] for n in reversed(names)]))
    a = compare_to_raters([], raters, fps=FPS, n_frames=N_FRAMES)
    b = compare_to_raters([], shuffled, fps=FPS, n_frames=N_FRAMES)
    assert a.human_range == b.human_range
    assert sorted(p.kappa for p in a.between_raters) == pytest.approx(
        sorted(p.kappa for p in b.between_raters)
    )


# ---------------------------------------------------------------- report plumbing


def test_report_is_json_serialisable():
    _, raters = rater_panel(3)
    rep = compare_to_raters(
        random_bouts(np.random.default_rng(9)), raters, fps=FPS, n_frames=N_FRAMES
    )
    d = rep.to_dict()
    text = json.dumps(d, allow_nan=False)
    back = json.loads(text)
    assert back["within_human_range"] == rep.within_human_range
    assert back["human_range"] == list(rep.human_range)
    assert back["vs_raters"][0]["kappa"] == rep.vs_raters[0].kappa
    assert back["vs_raters"][0]["table"] == [list(r) for r in rep.vs_raters[0].table]


def test_single_rater_has_no_human_range():
    rep = compare_to_raters([(0.0, 1.0)], {"ann": [(0.0, 1.0)]}, fps=FPS, n_frames=100)
    assert rep.between_raters == ()
    assert rep.human_range is None and rep.within_human_range is None
    json.dumps(rep.to_dict(), allow_nan=False)


def test_rater_cannot_be_named_instrument():
    with pytest.raises(ValueError):
        compare_to_raters([], {"instrument": []}, fps=FPS, n_frames=10)


def test_pairwise_agreement_is_the_human_block_of_the_report():
    _, raters = rater_panel(4)
    rep = compare_to_raters([], raters, fps=FPS, n_frames=N_FRAMES)
    assert pairwise_agreement(raters, fps=FPS, n_frames=N_FRAMES) == rep.between_raters


def test_bouts_from_events_round_trip_frames():
    events = [
        BehaviourEvent("grooming", 1, Span(25, 50), 0.9, "test"),
        BehaviourEvent("grooming", 2, Span(100, 125), 0.9, "test"),
        BehaviourEvent("supported_rear", 1, Span(0, 10), 0.9, "test"),
    ]
    assert bouts_from_events(events, "grooming", fps=25.0) == [(1.0, 2.0), (4.0, 5.0)]
    assert bouts_from_events(events, "grooming", fps=25.0, subject=1) == [(1.0, 2.0)]
    rep = compare_to_raters(
        bouts_from_events(events, "grooming", 25.0, subject=1),
        {"ann": [(1.0, 2.0)]},
        fps=25.0,
        n_frames=200,
    )
    assert rep.vs_raters[0].kappa == 1.0


# ---------------------------------------------------------------- dataset level


def dataset(n_videos=8, instrument=None, names=("ann", "bob", "cat"), permute=None):
    """Per-video reports; ``permute`` (an rng) reassigns rater names independently per video."""
    reports = []
    for v in range(n_videos):
        rng = np.random.default_rng(100 + v)
        truth = random_bouts(rng, rate_per_s=rng.uniform(0.02, 0.08))
        scored = [noisy_copy(rng, truth) for _ in names]
        pred = scored[0] if instrument is None else instrument(rng, truth)
        order = range(len(names)) if permute is None else permute.permutation(len(names))
        raters = {names[i]: scored[j] for i, j in zip(range(len(names)), order)}
        reports.append(compare_to_raters(pred, raters, fps=FPS, n_frames=N_FRAMES))
    return reports


def test_aggregate_count_agreement_matches_direct_computation():
    reports = dataset(instrument=lambda rng, truth: noisy_copy(rng, truth, drop=0.2))
    agg = aggregate(reports)
    assert agg.n_videos == 8
    inst = [r.instrument_count for r in reports]
    mean_rater = [np.mean(list(r.rater_counts.values())) for r in reports]
    q = agg.quantity("instrument", "count")
    assert q.r == pytest.approx(pearson_r(inst, mean_rater))
    assert q.icc_2_1 == pytest.approx(icc(np.column_stack([inst, mean_rater]), "2,1"))
    bob = [r.rater_counts["bob"] for r in reports]
    others = [np.mean([r.rater_counts["ann"], r.rater_counts["cat"]]) for r in reports]
    assert agg.quantity("bob", "count").r == pytest.approx(pearson_r(bob, others))
    assert {qa.who for qa in agg.quantities} == {"instrument", "ann", "bob", "cat"}


def test_aggregate_pools_frames_across_videos():
    reports = dataset()
    agg = aggregate(reports)
    pooled = {(p.a, p.b): p for p in agg.human_pairs}
    table = sum(
        np.array(next(x for x in r.between_raters if (x.a, x.b) == ("ann", "bob")).table)
        for r in reports
    )
    from kappa.eval.stats import cohen_kappa_from_table

    assert pooled[("ann", "bob")].kappa == pytest.approx(cohen_kappa_from_table(table))
    assert agg.within_human_range is True  # the instrument is a copy of rater "ann"
    lo, hi = agg.human_kappa_range
    assert lo <= hi
    json.dumps(agg.to_dict(), allow_nan=False)


def test_aggregate_random_instrument_is_flagged():
    agg = aggregate(dataset(instrument=lambda rng, truth: random_bouts(rng)))
    assert agg.within_human_range is False
    assert abs(agg.instrument_mean_kappa) < 0.1
    assert agg.quantity("instrument", "count").r < agg.human_range("count", "r")[0]


def test_aggregate_shuffled_rater_identities_keep_the_human_range():
    base = aggregate(dataset())
    renamed = aggregate(dataset(names=("zed", "amy", "kim")))
    assert renamed.human_kappa_range == pytest.approx(base.human_kappa_range)
    assert renamed.human_range("count", "r") == pytest.approx(base.human_range("count", "r"))
    assert renamed.within_human_range == base.within_human_range
    # Reassigning names independently per video leaves every per-video human range intact.
    for a, b in zip(dataset(), dataset(permute=np.random.default_rng(7))):
        assert a.human_range == pytest.approx(b.human_range)


def test_aggregate_needs_reports():
    with pytest.raises(ValueError):
        aggregate([])
    assert isinstance(dataset(n_videos=1)[0], AgreementReport)
    one = aggregate(dataset(n_videos=1))
    assert math.isnan(one.quantity("instrument", "count").r)
