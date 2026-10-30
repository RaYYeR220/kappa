import math

import pytest

from kappa.eval.mot import box_iou, clear_mot, id_switches_per_100k_frames, idf1


def box(x, y, size=10.0):
    return (x, y, x + size, y + size)


A = box(0, 0)  # two objects far apart so every IoU is 0 or 1
B = box(100, 100)


def test_box_iou():
    assert box_iou(A, A) == 1.0
    assert box_iou(A, B) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_perfect_tracking():
    gt = [{1: A, 2: B} for _ in range(10)]
    pred = [{7: A, 8: B} for _ in range(10)]
    mot = clear_mot(gt, pred)
    assert (mot.matches, mot.fp, mot.fn, mot.id_switches) == (20, 0, 0, 0)
    assert mot.mota == 1.0 and mot.motp == 1.0
    ident = idf1(gt, pred)
    assert (ident.idtp, ident.idfp, ident.idfn) == (20, 0, 0)
    assert ident.idf1 == ident.idp == ident.idr == 1.0
    assert ident.assignment == {1: 7, 2: 8}


def test_mid_sequence_swap():
    # Tracker ids 7 and 8 trade objects at frame 5 of 10.
    gt = [{1: A, 2: B} for _ in range(10)]
    pred = [{7: A, 8: B} if t < 5 else {7: B, 8: A} for t in range(10)]
    mot = clear_mot(gt, pred)
    assert (mot.fp, mot.fn, mot.id_switches) == (0, 0, 2)
    assert mot.mota == pytest.approx(1 - 2 / 20)
    ident = idf1(gt, pred)
    assert (ident.idtp, ident.idfp, ident.idfn) == (10, 10, 10)
    assert ident.idf1 == pytest.approx(0.5)
    assert ident.idp == pytest.approx(0.5) and ident.idr == pytest.approx(0.5)


def test_early_swap_keeps_the_longer_identity():
    gt = [{1: A, 2: B} for _ in range(10)]
    pred = [{7: A, 8: B} if t < 3 else {7: B, 8: A} for t in range(10)]
    ident = idf1(gt, pred)
    assert ident.assignment == {1: 8, 2: 7}
    assert (ident.idtp, ident.idfp, ident.idfn) == (14, 6, 6)
    assert ident.idf1 == pytest.approx(0.7)
    assert clear_mot(gt, pred).id_switches == 2


def test_misses_and_false_positives():
    gt = [{1: A} for _ in range(10)]
    pred = [{} if t < 2 else {7: A} for t in range(10)]
    pred[5] = {7: A, 9: B}
    mot = clear_mot(gt, pred)
    assert (mot.n_gt, mot.matches, mot.fn, mot.fp, mot.id_switches) == (10, 8, 2, 1, 0)
    assert mot.mota == pytest.approx(0.7)
    ident = idf1(gt, pred)
    assert (ident.idtp, ident.idfp, ident.idfn) == (8, 1, 2)
    assert ident.idf1 == pytest.approx(16 / 19)


def test_a_still_valid_correspondence_is_not_broken():
    # Hypothesis 8 drifts onto object 1 with a better IoU than 7, but 7 still overlaps >= 0.5.
    gt = [{1: box(0, 0)} for _ in range(4)]
    pred = [
        {7: box(0, 0)},
        {7: box(2, 0), 8: box(0, 0)},
        {7: box(2, 0), 8: box(0, 0)},
        {7: box(0, 0)},
    ]
    mot = clear_mot(gt, pred)
    assert mot.id_switches == 0
    assert mot.fp == 2


def test_reacquiring_after_a_gap():
    gt = [{1: A}] * 6
    same = [{7: A}, {7: A}, {}, {}, {7: A}, {7: A}]
    other = [{7: A}, {7: A}, {}, {}, {9: A}, {9: A}]
    assert clear_mot(gt, same).id_switches == 0
    assert clear_mot(gt, other).id_switches == 1


def test_iou_threshold_gates_matches():
    gt = [{1: (0, 0, 10, 10)}]
    pred = [{7: (5, 0, 15, 10)}]  # IoU 1/3
    mot = clear_mot(gt, pred)
    assert (mot.matches, mot.fp, mot.fn) == (0, 1, 1)
    assert math.isnan(mot.motp)
    assert clear_mot(gt, pred, iou_threshold=0.3).matches == 1


def test_empty_and_mismatched_inputs():
    mot = clear_mot([{}, {}], [{}, {}])
    assert mot.n_gt == 0 and math.isnan(mot.mota)
    assert idf1([{}], [{}]).idf1 == 1.0
    with pytest.raises(ValueError):
        clear_mot([{1: A}], [])
    with pytest.raises(ValueError):
        idf1([{1: A}], [])


def test_id_switch_rate():
    assert id_switches_per_100k_frames(2, 10) == pytest.approx(20_000)
    assert id_switches_per_100k_frames(3, 300_000) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        id_switches_per_100k_frames(1, 0)
