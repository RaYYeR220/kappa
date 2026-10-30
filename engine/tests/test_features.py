import numpy as np
import pytest

from kappa.features import mask_features
from sam_synth import ellipse_mask


def test_empty_mask_has_no_features():
    assert mask_features(np.zeros((5, 5), np.uint8), 0, 0) is None


def test_centroid_is_in_image_coordinates():
    mask = np.ones((4, 6), np.uint8)
    f = mask_features(mask, 100, 50)
    assert f.area_px == 24
    assert f.cx == pytest.approx(100 + 2.5)
    assert f.cy == pytest.approx(50 + 1.5)


def test_ellipse_axes_and_orientation():
    f = mask_features(ellipse_mask(20, 60), 0, 0)
    assert f.major_px == pytest.approx(60, rel=0.05)
    assert f.minor_px == pytest.approx(20, rel=0.08)
    assert abs(f.angle_rad) < 0.01
    g = mask_features(ellipse_mask(60, 20), 0, 0)
    assert abs(abs(g.angle_rad) - np.pi / 2) < 0.01


def test_zone_fraction_uses_offset_mask():
    raster = np.zeros((100, 100), np.uint8)
    raster[:, 50:] = 1
    mask = np.ones((10, 10), np.uint8)
    assert mask_features(mask, 45, 10, {"right": raster}).zone_fraction["right"] == pytest.approx(0.5)
    assert mask_features(mask, 60, 10, {"right": raster}).zone_fraction["right"] == pytest.approx(1.0)
    assert mask_features(mask, 95, 10, {"right": raster}).zone_fraction["right"] == pytest.approx(0.5)
