import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from synth import FPS, protocol, track

from kappa.metrics import compute_metrics
from kappa.model import Span, Status

BIG = {"shape": "rect", "width_cm": 1000, "height_cm": 1000}
MID = 500.0

offset = st.floats(-50, 50, allow_nan=False)
point = st.one_of(st.tuples(offset, offset), st.just((math.nan, math.nan)))
paths = st.lists(point, min_size=2, max_size=80).map(lambda ps: np.array(ps, dtype=float).T)

settings.register_profile("kappa", deadline=None, max_examples=150)
settings.load_profile("kappa")


def measure(metrics, x, y, *, smoothing=None, zones=(), arena=BIG, **proto):
    p = protocol(metrics, smoothing=smoothing, zones=list(zones), arena=arena, **proto)
    return compute_metrics(p, {1: track(x, y)}, [], [], Span(0, len(x)))


def distance(x, y, smoothing=None) -> float:
    (r,) = measure([{"id": "d", "kind": "distance"}], x, y, smoothing=smoothing)
    return r.value if r.status is Status.VERIFIED else 0.0


@given(paths, offset, offset, st.sampled_from(["none", "median", "savgol"]))
def test_distance_is_invariant_under_translation(path, dx, dy, method):
    x, y = MID + path[0], MID + path[1]
    smoothing = {"method": method}
    assert distance(x + 4 * dx, y + 4 * dy, smoothing) == pytest.approx(
        distance(x, y, smoothing), rel=1e-9, abs=1e-6
    )


@given(paths, st.floats(0, 2 * math.pi), st.sampled_from(["none", "savgol"]))
def test_distance_is_invariant_under_rotation(path, theta, method):
    """Median smoothing filters each axis separately, so it is not rotation-equivariant."""
    c, s = math.cos(theta), math.sin(theta)
    x, y = MID + path[0], MID + path[1]
    rx, ry = MID + c * path[0] - s * path[1], MID + s * path[0] + c * path[1]
    smoothing = {"method": method}
    assert distance(rx, ry, smoothing) == pytest.approx(
        distance(x, y, smoothing), rel=1e-9, abs=1e-6
    )


@given(paths, st.floats(0.01, 30), st.sampled_from(["none", "median", "savgol"]))
def test_min_step_never_adds_distance(path, min_step, method):
    """The anchor method credits a polyline through a subset of the path's points, so by the
    triangle inequality it never exceeds the unsuppressed distance. It is not monotone between
    two positive steps: on 0 → 1 → 1.6 a 1.0 cm step credits 1.0 cm but a 1.5 cm step 1.6 cm."""
    x, y = MID + path[0], MID + path[1]
    damped = distance(x, y, {"method": method, "min_step_cm": min_step})
    assert 0.0 <= damped <= distance(x, y, {"method": method}) + 1e-9


@given(paths, st.floats(0, 3), st.integers(1, 30), st.sampled_from(["none", "median"]))
def test_binned_distance_sums_to_the_whole(path, min_step, bin_frames, method):
    x, y = MID + path[0], MID + path[1]
    assume(np.isfinite(x).any())
    metrics = [{"id": "d", "kind": "distance", "bin_s": bin_frames / FPS}]
    rs = measure(metrics, x, y, smoothing={"method": method, "min_step_cm": min_step})
    whole = next(r for r in rs if r.bin is None)
    parts = [r.value for r in rs if r.bin is not None and r.status is Status.VERIFIED]
    assert sum(r.bin.length for r in rs if r.bin is not None) == len(x)
    assert sum(parts) == pytest.approx(whole.value, abs=1e-9)


ZONES = [
    {"id": "z", "kind": "center_fraction", "fraction": 0.3},
    {"id": "z", "kind": "periphery_band", "band_cm": 8.0},
    {"id": "z", "kind": "rect_cm", "rect_cm": [10.0, -5.0, 30.0, 25.0]},
]
arena_point = st.one_of(
    st.tuples(st.floats(-5, 50, allow_nan=False), st.floats(-5, 50, allow_nan=False)),
    st.just((math.nan, math.nan)),
)


@given(
    st.lists(arena_point, min_size=1, max_size=120).map(lambda ps: np.array(ps, dtype=float).T),
    st.sampled_from(ZONES),
    st.sampled_from(["rect", "circle"]),
)
def test_time_in_zone_never_exceeds_time_observed(path, zone, shape):
    x, y = path
    metrics = [
        {"id": "t", "kind": "time_in_zone", "zone": "z"},
        {"id": "f", "kind": "fraction_time_in_zone", "zone": "z"},
    ]
    arena = {"shape": shape, "width_cm": 45, "height_cm": 45}
    t, f = measure(metrics, x, y, zones=[zone], arena=arena)
    observed = np.isfinite(x).sum()
    if observed == 0:
        assert t.status is Status.REFUSED and f.status is Status.REFUSED
        return
    assert 0.0 <= t.value <= observed / FPS + 1e-12
    assert 0.0 <= f.value <= 100.0
    assert f.value == pytest.approx(t.value * FPS / observed * 100)
