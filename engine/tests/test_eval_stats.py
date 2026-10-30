import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from kappa.eval.stats import (
    bland_altman,
    bout_f1,
    bout_match,
    bout_spans,
    bouts_to_frames,
    cohen_kappa,
    cohen_kappa_from_table,
    icc,
    pearson_r,
    temporal_iou,
)
from kappa.model import Span

# ---------------------------------------------------------------- bouts -> frames


def test_frame_is_on_when_its_midpoint_is_inside_the_bout():
    # fps 10: frame f has midpoint (f + 0.5) / 10 s. Midpoints 0.25, 0.35, 0.45 lie in [0.2, 0.5).
    mask = bouts_to_frames([(0.2, 0.5)], fps=10, n_frames=10)
    assert mask.dtype == bool and mask.shape == (10,)
    assert np.flatnonzero(mask).tolist() == [2, 3, 4]


def test_midpoint_on_the_onset_is_inside_and_on_the_offset_is_outside():
    mask = bouts_to_frames([(0.25, 0.45)], fps=10, n_frames=10)
    assert np.flatnonzero(mask).tolist() == [2, 3]


def test_bouts_are_clipped_to_the_frame_range():
    mask = bouts_to_frames([(-1.0, 0.3), (0.8, 5.0), (7.0, 9.0)], fps=10, n_frames=10)
    assert np.flatnonzero(mask).tolist() == [0, 1, 2, 8, 9]


def test_offset_shifts_the_label_clock():
    mask = bouts_to_frames([(1.2, 1.5)], fps=10, n_frames=10, offset_s=1.0)
    assert np.flatnonzero(mask).tolist() == [2, 3, 4]


def test_a_bout_shorter_than_a_frame_can_vanish():
    assert not bouts_to_frames([(0.21, 0.24)], fps=10, n_frames=10).any()


def test_overlapping_bouts_merge_in_the_mask_but_stay_separate_spans():
    bouts = [(0.0, 0.5), (0.3, 0.8)]
    assert bouts_to_frames(bouts, fps=10, n_frames=10).sum() == 8
    assert bout_spans(bouts, fps=10, n_frames=10) == [Span(0, 5), Span(3, 8)]


def test_bouts_to_frames_validates_input():
    with pytest.raises(ValueError):
        bouts_to_frames([(0.5, 0.4)], fps=10, n_frames=10)
    with pytest.raises(ValueError):
        bouts_to_frames([], fps=0, n_frames=10)
    with pytest.raises(ValueError):
        bouts_to_frames([], fps=10, n_frames=-1)


@given(
    st.lists(st.tuples(st.integers(0, 500), st.integers(0, 60)), max_size=10),
    st.sampled_from([7.5, 25.0, 29.97, 30.0, 60.0]),
)
def test_frame_spans_round_trip_through_seconds(raw, fps):
    spans = [Span(s, s + d) for s, d in raw if d > 0]
    seconds = [(sp.start / fps, sp.end / fps) for sp in spans]
    assert bout_spans(seconds, fps=fps, n_frames=600) == spans


# ---------------------------------------------------------------- Cohen's kappa


def test_kappa_textbook_example():
    # 50 items; both yes 20, A yes/B no 5, A no/B yes 10, both no 15: p_o = 0.7, p_e = 0.5.
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa_from_table(np.array([[15, 10], [5, 20]])) == pytest.approx(0.4)


def test_kappa_categorical():
    # agreement 4/6, uniform marginals -> p_e = 1/3, kappa = (2/3 - 1/3) / (2/3).
    assert cohen_kappa([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 2, 1]) == pytest.approx(0.5)
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_kappa_extremes():
    a = np.array([True, False] * 50)
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, ~a) == pytest.approx(-1.0)


def test_kappa_degenerate_expected_agreement():
    # Both raters use one identical category everywhere: p_e = 1, defined as perfect agreement.
    assert cohen_kappa(np.zeros(10, bool), np.zeros(10, bool)) == 1.0
    assert cohen_kappa_from_table(np.array([[10, 0], [0, 0]])) == 1.0


def test_one_rater_never_scoring_gives_zero_kappa():
    b = np.zeros(100, bool)
    b[10] = True
    assert cohen_kappa(np.zeros(100, bool), b) == 0.0


def test_kappa_validates_input():
    with pytest.raises(ValueError):
        cohen_kappa([1, 0], [1])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


@given(st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=200))
def test_kappa_is_symmetric_and_bounded(pairs):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    k = cohen_kappa(a, b)
    assert -1.0 - 1e-12 <= k <= 1.0 + 1e-12
    assert k == pytest.approx(cohen_kappa(b, a))


# ---------------------------------------------------------------- correlation, ICC, Bland-Altman


def test_pearson_matches_numpy_and_flags_constant_input():
    x = [1.0, 2.0, 4.0, 3.0, 7.0]
    y = [2.0, 1.0, 5.0, 4.0, 9.0]
    assert pearson_r(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert math.isnan(pearson_r([1, 1, 1], [1, 2, 3]))
    with pytest.raises(ValueError):
        pearson_r([1, 2], [1, 2, 3])


SHROUT_FLEISS_TABLE_2 = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


def test_icc_reproduces_shrout_fleiss_worked_example():
    assert round(icc(SHROUT_FLEISS_TABLE_2, kind="2,1"), 2) == 0.29
    assert round(icc(SHROUT_FLEISS_TABLE_2, kind="3,1"), 2) == 0.71


def test_icc_absolute_agreement_penalises_a_rater_offset_consistency_does_not():
    base = np.array([1.0, 3.0, 2.0, 6.0, 5.0])
    ratings = np.column_stack([base, base + 2.0])
    assert icc(ratings, kind="3,1") == pytest.approx(1.0)
    assert icc(ratings, kind="2,1") < 0.9
    same = np.column_stack([base, base, base])
    assert icc(same, kind="2,1") == pytest.approx(1.0)


def test_icc_is_invariant_to_rater_order():
    perm = SHROUT_FLEISS_TABLE_2[:, [2, 0, 3, 1]]
    for kind in ("2,1", "3,1"):
        assert icc(perm, kind=kind) == pytest.approx(icc(SHROUT_FLEISS_TABLE_2, kind=kind))


def test_icc_validates_input():
    with pytest.raises(ValueError):
        icc(np.ones((1, 3)))
    with pytest.raises(ValueError):
        icc(np.ones((3, 1)))
    with pytest.raises(ValueError):
        icc(SHROUT_FLEISS_TABLE_2, kind="1,1")


def test_bland_altman_limits_of_agreement():
    mean_diff, lower, upper = bland_altman([1, 2, 3, 4], [0, 2, 2, 5])
    sd = np.std([1, 0, 1, -1], ddof=1)
    assert mean_diff == pytest.approx(0.25)
    assert lower == pytest.approx(0.25 - 1.96 * sd)
    assert upper == pytest.approx(0.25 + 1.96 * sd)


# ---------------------------------------------------------------- bout matching


def test_temporal_iou():
    assert temporal_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert temporal_iou((0, 10), (10, 20)) == 0.0
    assert temporal_iou(Span(0, 10), (0, 10)) == 1.0
    assert temporal_iou((3, 3), (3, 3)) == 0.0


def test_bout_match_counts_hits_misses_and_false_alarms():
    tp, fp, fn, matches = bout_match([(0, 10), (20, 30)], [(1, 10), (50, 60)])
    assert (tp, fp, fn) == (1, 1, 1)
    assert [(m.pred, m.ref) for m in matches] == [(0, 0)]
    assert matches[0].iou == pytest.approx(0.9)


def test_bout_match_is_one_to_one():
    tp, fp, fn, matches = bout_match([(0, 10)], [(0, 4), (5, 10)])
    assert (tp, fp, fn) == (1, 0, 1)
    assert matches[0].ref == 1


def test_bout_match_is_globally_optimal_not_greedy():
    # Greedy would take the single best pair (A, X: IoU 0.9) and strand B and Y.
    pred = [(0, 10), (2, 12)]  # A, B
    ref = [(1, 10), (0, 5)]  # X, Y
    tp, fp, fn, matches = bout_match(pred, ref)
    assert (tp, fp, fn) == (2, 0, 0)
    assert {(m.pred, m.ref) for m in matches} == {(0, 1), (1, 0)}


def test_bout_match_threshold_is_inclusive():
    assert bout_match([(0, 3)], [(0, 10)], min_iou=0.3).tp == 1
    assert bout_match([(0, 3)], [(0, 10)], min_iou=0.31).tp == 0


def test_bout_f1_edge_cases():
    assert bout_f1([], [(0, 1)]) == 0.0
    assert bout_f1([(0, 1)], []) == 0.0
    assert bout_f1([], []) == 1.0
    assert bout_f1([(0, 10), (20, 30)], [(1, 10), (50, 60)]) == pytest.approx(0.5)


@given(
    st.lists(st.tuples(st.integers(0, 100), st.integers(1, 20)), max_size=8),
    st.lists(st.tuples(st.integers(0, 100), st.integers(1, 20)), max_size=8),
)
def test_bout_match_bookkeeping(p, r):
    pred = [(s, s + d) for s, d in p]
    ref = [(s, s + d) for s, d in r]
    res = bout_match(pred, ref)
    assert res.tp + res.fp == len(pred)
    assert res.tp + res.fn == len(ref)
    assert len({m.pred for m in res.matches}) == len({m.ref for m in res.matches}) == res.tp
    assert all(m.iou >= 0.3 for m in res.matches)
