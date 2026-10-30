import cv2
import numpy as np
import pytest

from kappa.geometry import Calibration, inside, project, zone_raster
from kappa.model import SubjectTrack
from kappa.protocol import ArenaSpec, ZoneSpec

SKEWED_PX = ((100.0, 80.0), (560.0, 95.0), (545.0, 520.0), (110.0, 505.0))
SQUARE_PX = ((50.0, 40.0), (500.0, 40.0), (500.0, 490.0), (50.0, 490.0))
RIM_PX = ((100.0, 300.0), (300.0, 100.0), (500.0, 300.0), (300.0, 500.0))


def rect(corners=SKEWED_PX, w=45.0, h=45.0):
    return ArenaSpec(shape="rect", width_cm=w, height_cm=h, corners_px=corners)


def circle(rim=RIM_PX, d=40.0):
    return ArenaSpec(shape="circle", width_cm=d, height_cm=d, corners_px=rim)


def uniform(arena, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, arena.width_cm, n), rng.uniform(0, arena.height_cm, n)


def test_rect_corners_map_to_arena_corners():
    cal = Calibration(rect())
    x, y = cal.to_cm([p[0] for p in SKEWED_PX], [p[1] for p in SKEWED_PX])
    assert np.allclose(x, [0, 45, 45, 0], atol=1e-4)
    assert np.allclose(y, [0, 0, 45, 45], atol=1e-4)


def test_circle_rim_points_map_to_rim():
    cal = Calibration(circle())
    x, y = cal.to_cm([p[0] for p in RIM_PX], [p[1] for p in RIM_PX])
    assert np.allclose(x, [0, 20, 40, 20], atol=1e-4)
    assert np.allclose(y, [20, 0, 20, 40], atol=1e-4)


@pytest.mark.parametrize("arena", [rect(), circle()])
def test_homography_round_trip(arena):
    cal = Calibration(arena)
    rng = np.random.default_rng(1)
    px, py = rng.uniform(0, 640, 500), rng.uniform(0, 480, 500)
    back_x, back_y = cal.to_px(*cal.to_cm(px, py))
    assert np.allclose(back_x, px, atol=1e-6)
    assert np.allclose(back_y, py, atol=1e-6)


def test_nan_in_nan_out():
    cal = Calibration(rect())
    x, y = cal.to_cm(np.array([np.nan, 300.0]), np.array([np.nan, 300.0]))
    assert np.isnan(x[0]) and np.isnan(y[0])
    assert np.isfinite(x[1]) and np.isfinite(y[1])


def test_px_per_cm_of_an_unskewed_view():
    assert Calibration(rect(SQUARE_PX)).px_per_cm == pytest.approx(10.0)
    assert Calibration(circle()).px_per_cm == pytest.approx(10.0)


def test_calibration_needs_corners():
    with pytest.raises(ValueError, match="corners_px"):
        Calibration(ArenaSpec(shape="rect", width_cm=45, height_cm=45))


def test_corners_out_of_order_are_rejected():
    tl, tr, br, bl = SKEWED_PX
    with pytest.raises(ValueError, match="order"):
        Calibration(rect((tl, tr, bl, br)))


def test_project_fills_cm_and_leaves_input_alone():
    track = SubjectTrack.empty(1, 3, 25.0)
    track.present[:2] = True
    track.cx[:2] = [50.0, 500.0]
    track.cy[:2] = [40.0, 490.0]
    projected = project(track, Calibration(rect(SQUARE_PX)))
    assert track.x_cm is None and track.y_cm is None
    assert np.allclose(projected.x_cm[:2], [0, 45])
    assert np.allclose(projected.y_cm[:2], [0, 45])
    assert np.isnan(projected.x_cm[2]) and np.isnan(projected.y_cm[2])
    projected.cx[0] = -1.0
    assert track.cx[0] == 50.0


@pytest.mark.parametrize("arena", [rect(w=60, h=40), circle()])
@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5])
def test_centre_zone_covers_its_fraction_of_the_arena(arena, fraction):
    zone = ZoneSpec(id="center", kind="center_fraction", fraction=fraction)
    x, y = uniform(arena)
    in_arena = inside(ZoneSpec(id="all", kind="periphery_band", band_cm=1e6), arena, x, y)
    in_centre = inside(zone, arena, x, y)
    assert not (in_centre & ~in_arena).any()
    assert in_centre.sum() / in_arena.sum() == pytest.approx(fraction, abs=0.005)


def test_band_and_centre_partition_a_square_arena_when_band_is_the_inset():
    arena = rect()
    centre = ZoneSpec(id="center", kind="center_fraction", fraction=0.25)
    band = ZoneSpec(id="wall", kind="periphery_band", band_cm=45 * (1 - 0.5) / 2)
    x, y = uniform(arena)
    a, b = inside(centre, arena, x, y), inside(band, arena, x, y)
    assert not (a & b).any()
    assert (a | b).all()


def test_circle_band_is_within_band_of_the_rim():
    arena = circle()
    band = ZoneSpec(id="wall", kind="periphery_band", band_cm=5.0)
    x = np.array([20.0, 20.0, 20.0, 1.0, -1.0])
    y = np.array([20.0, 5.5, 4.5, 20.0, 20.0])
    assert inside(band, arena, x, y).tolist() == [False, False, True, True, False]


def test_rect_zone_is_clipped_to_the_arena():
    arena = rect()
    zone = ZoneSpec(id="corner", kind="rect_cm", rect_cm=(-10.0, -10.0, 10.0, 10.0))
    x = np.array([5.0, -5.0, 15.0, 10.0])
    y = np.array([5.0, 5.0, 5.0, 10.0])
    assert inside(zone, arena, x, y).tolist() == [True, False, False, True]


@pytest.mark.parametrize(
    "zone",
    [
        ZoneSpec(id="center", kind="center_fraction", fraction=0.9),
        ZoneSpec(id="wall", kind="periphery_band", band_cm=100.0),
        ZoneSpec(id="box", kind="rect_cm", rect_cm=(-100.0, -100.0, 100.0, 100.0)),
    ],
)
@pytest.mark.parametrize("arena", [rect(), circle()])
def test_points_outside_the_arena_are_in_no_zone(zone, arena):
    x = np.array([-0.5, arena.width_cm + 0.5, arena.width_cm / 2, 0.3, np.nan])
    y = np.array([arena.height_cm / 2, arena.height_cm / 2, -0.5, 0.3, 1.0])
    expected_outside = [True, True, True, arena.shape == "circle", True]
    assert not inside(zone, arena, x, y)[np.array(expected_outside)].any()


def _pixel_centres_in_zone(zone, arena, cal, width, height):
    yy, xx = np.mgrid[0:height, 0:width]
    return inside(zone, arena, *cal.to_cm(xx.astype(float), yy.astype(float)))


def _edge_pixels(mask):
    kernel = np.ones((3, 3), np.uint8)
    m = mask.astype(np.uint8)
    return cv2.dilate(m, kernel) != cv2.erode(m, kernel)


@pytest.mark.parametrize(
    "zone",
    [
        ZoneSpec(id="center", kind="center_fraction", fraction=0.25),
        ZoneSpec(id="wall", kind="periphery_band", band_cm=7.0),
        ZoneSpec(id="box", kind="rect_cm", rect_cm=(30.0, -5.0, 50.0, 12.0)),
    ],
)
@pytest.mark.parametrize("arena", [rect(), circle()])
def test_raster_agrees_with_the_analytic_zone_away_from_its_edge(zone, arena):
    cal = Calibration(arena)
    raster = zone_raster(zone, arena, cal, 640, 600)
    assert raster.dtype == np.uint8 and raster.shape == (600, 640)
    assert set(np.unique(raster).tolist()) == {0, 1}
    analytic = _pixel_centres_in_zone(zone, arena, cal, 640, 600)
    disagree = raster.astype(bool) != analytic
    assert not (disagree & ~_edge_pixels(analytic)).any()


def test_raster_areas_follow_the_zone_definition_in_an_unskewed_view():
    arena, cal = rect(SQUARE_PX), Calibration(rect(SQUARE_PX))
    full = zone_raster(
        ZoneSpec(id="all", kind="periphery_band", band_cm=100.0), arena, cal, 600, 600
    )
    centre = zone_raster(
        ZoneSpec(id="center", kind="center_fraction", fraction=0.25), arena, cal, 600, 600
    )
    band = zone_raster(
        ZoneSpec(id="wall", kind="periphery_band", band_cm=11.25), arena, cal, 600, 600
    )
    assert int(full.sum()) == pytest.approx(450 * 450, rel=0.01)
    assert int(centre.sum()) / int(full.sum()) == pytest.approx(0.25, abs=0.01)
    assert not (centre & band).any()
    assert ((centre | band) == full).all()


def test_circle_raster_area():
    arena = circle()
    raster = zone_raster(
        ZoneSpec(id="all", kind="periphery_band", band_cm=100.0),
        arena,
        Calibration(arena),
        600,
        600,
    )
    assert int(raster.sum()) == pytest.approx(np.pi * 200**2, rel=0.01)
