"""The ready protocols are valid, say what they still need, and measure what they claim."""

import pytest

from kappa import library
from kappa.protocol import AssayProtocol


def test_every_library_protocol_validates():
    assert library.names() == ["elevated_plus_maze", "open_field", "zebrafish_novel_tank"]
    for name in library.names():
        p = AssayProtocol.model_validate(library.load(name))
        assert p.arena.corners_px is None, "corners depend on the camera"
        assert p.notes and p.title


def test_open_field():
    p = library.protocol("open_field")
    assert (p.arena.width_cm, p.arena.height_cm, p.subject.expected_count) == (45, 45, 1)
    zone = next(z for z in p.zones if z.id == "center")
    assert (zone.kind, zone.fraction) == ("center_fraction", 0.25)
    kinds = {m.kind: m for m in p.metrics}
    assert kinds["distance"].bin_s == 300
    assert {"time_in_zone", "zone_entries", "latency_to_zone", "immobility_time", "behaviour_count"} <= set(kinds)
    assert kinds["behaviour_count"].behaviour == "rearing"
    assert p.behaviours[0].detector == "rearing_topdown"


def test_zebrafish_novel_tank():
    p = library.protocol("zebrafish_novel_tank")
    assert (p.subject.concept, p.subject.expected_count) == ("zebrafish", 5)
    assert {m.kind for m in p.metrics} == {"distance", "time_in_zone", "mean_inter_subject_distance"}
    assert p.zones[0].kind == "periphery_band"
    assert any("depth" in r.reason for r in p.refused)


def test_elevated_plus_maze_zones_tile_the_cross():
    p = library.protocol("elevated_plus_maze")
    side = p.arena.width_cm
    rects = {z.id: z.rect_cm for z in p.zones}
    assert all(z.kind == "rect_cm" for z in p.zones)
    assert set(rects) == {"center", "open_left", "open_right", "closed_top", "closed_bottom"}
    for x0, y0, x1, y1 in rects.values():
        assert 0 <= x0 < x1 <= side and 0 <= y0 < y1 <= side
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects.values())
    assert area == pytest.approx(2 * 30 * 5 + 2 * 30 * 5 + 5 * 5)
    assert "corners_px" in p.notes and "rect_cm" in p.notes


def test_catalogue_names_what_is_missing():
    entries = library.catalogue()
    assert [e["needs"] for e in entries] == [["arena.corners_px"]] * 3
    assert all(e["sha256"] == AssayProtocol.model_validate(e["protocol"]).content_hash() for e in entries)


def test_unknown_protocol():
    with pytest.raises(KeyError):
        library.load("forced_swim")
