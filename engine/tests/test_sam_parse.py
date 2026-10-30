import numpy as np
import pytest

from kappa.sam.parse import parse_output
from sam_synth import ellipse_mask, line, record


def _clip():
    m = ellipse_mask(12, 30)
    lines = [
        line(0, [record(0, 100, 200, m, 928, 576, 0.97), record(3, 400, 300, m, 928, 576, 0.88)]),
        line(1, [record(0, 104, 201, m, 928, 576, 0.95)]),
        line(3, [record(3, 410, 300, m, 928, 576, 0.81), record(0, 108, 202, m, 928, 576, 0.6)]),
    ]
    return "\n".join(lines) + "\n"


def test_parse_video_output_into_detections():
    out = parse_output(_clip(), workers=1)
    assert (out.width, out.height) == (928, 576)
    keys = [(d.frame, d.track_id) for d in out.detections]
    assert keys == [(0, 0), (0, 3), (1, 0), (3, 0), (3, 3)]
    first = out.detections[0]
    assert first.confidence == pytest.approx(0.97)
    assert first.box == (100.0, 200.0, 130.0, 212.0)  # right/bottom exclusive
    assert first.cx == pytest.approx(100 + 14.5, abs=0.01)
    assert first.cy == pytest.approx(200 + 5.5, abs=0.01)
    assert not out.diagnostics


def test_parallel_decoding_matches_serial():
    raw = _clip()
    serial = parse_output(raw, workers=1).detections
    parallel = parse_output(raw, workers=2).detections
    assert serial == parallel


def test_zone_fraction_is_computed_per_detection():
    raster = np.zeros((576, 928), np.uint8)
    raster[:, :300] = 1
    out = parse_output(_clip(), zone_rasters={"left": raster}, workers=1)
    by_track = {d.track_id: d.zone_fraction["left"] for d in out.detections if d.frame == 0}
    assert by_track == {0: 1.0, 3: 0.0}


def test_empty_output_is_not_an_error():
    out = parse_output("", workers=1)
    assert out.detections == []
