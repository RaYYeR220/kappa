import json

import pytest

from kappa.agents.protocol_agent import draft_protocol, locate_arena, output_text


def _resp(obj):
    return {"output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": json.dumps(obj)}]}]}


GOOD = {
    "title": "Open field: locomotion and anxiety-like behaviour",
    "subject_concept": "mouse",
    "subject_label": "mouse",
    "expected_count": 1,
    "arena_shape": "rect",
    "arena_width_cm": 45,
    "arena_height_cm": 45,
    "zones": [{"id": "center", "kind": "center_fraction", "fraction": 0.25, "band_cm": None, "rect_cm": None}],
    "entry_mode": "body_fraction",
    "metrics": [
        {"id": "distance", "kind": "distance", "zone": None, "behaviour": None, "bin_s": 300},
        {"id": "center_time", "kind": "time_in_zone", "zone": "center", "behaviour": None, "bin_s": None},
    ],
    "behaviours": [],
    "max_speed_cm_s": 120,
    "refused": [{"request": "heart rate", "reason": "Not observable from a top-down video."}],
    "assumptions": ["Arena is 45 x 45 cm."],
    "notes": "Centre zone = central 25% of the floor.",
}
CORNERS = {"points": [
    {"object_name": "the top-left corner of the arena floor", "x": 350, "y": 260},
    {"object_name": "the top-right corner of the arena floor", "x": 760, "y": 260},
    {"object_name": "the bottom-right corner of the arena floor", "x": 760, "y": 820},
    {"object_name": "the bottom-left corner of the arena floor", "x": 350, "y": 1003},
]}


class FakeApi:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def create(self, body):
        self.bodies.append(body)
        return _resp(self.replies.pop(0))


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "k.jpg"
    path.write_bytes(b"\xff\xd8fakejpeg")
    return path


def test_output_text_skips_reasoning():
    assert output_text(_resp({"a": 1})) == '{"a": 1}'


def test_draft_builds_validated_protocol_with_refusals(frame):
    api = FakeApi([GOOD, CORNERS])
    draft = draft_protocol(api, "How far does it run, how long in the centre, and its heart rate?", [frame], 928, 576)
    p = draft.protocol
    assert p is not None
    assert p.subject.concept == "mouse"
    assert [m.id for m in p.metrics] == ["distance", "center_time"]
    assert p.refused[0].request == "heart rate"
    assert p.entry_rule.mode == "body_fraction"
    assert p.arena.corners_px[0] == pytest.approx((350 * 928 / 1000, 260 * 576 / 1000))
    assert p.arena.corners_px[3][1] == 575  # clamped into the frame
    assert "Arena is 45 x 45 cm." in draft.assumptions
    first = api.bodies[0]
    assert first["model"] == "muse-spark-1.3"
    assert first["text"]["format"]["type"] == "json_schema"
    assert any(part["type"] == "input_image" for part in first["input"][0]["content"])


def test_invalid_draft_is_repaired_once(frame):
    bad = dict(GOOD, metrics=[{"id": "center_time", "kind": "time_in_zone", "zone": "nowhere", "behaviour": None, "bin_s": None}])
    api = FakeApi([bad, CORNERS, GOOD])
    draft = draft_protocol(api, "q", [frame], 928, 576)
    assert draft.protocol is not None
    assert len(draft.errors) == 1
    assert "failed validation" in api.bodies[2]["input"][0]["content"][-1]["text"]


def test_unrepairable_draft_returns_no_protocol(frame):
    bad = dict(GOOD, metrics=[{"id": "x", "kind": "time_in_zone", "zone": "nowhere", "behaviour": None, "bin_s": None}])
    draft = draft_protocol(FakeApi([bad, CORNERS, bad]), "q", [frame], 928, 576)
    assert draft.protocol is None and len(draft.errors) == 2


def test_missing_corner_means_no_calibration(frame):
    api = FakeApi([{"points": CORNERS["points"][:3]}])
    assert locate_arena(api, frame, "rect", 928, 576) is None
    draft = draft_protocol(FakeApi([GOOD, {"points": []}]), "q", [frame], 928, 576)
    assert draft.protocol.arena.corners_px is None
    assert any("mark them" in a for a in draft.assumptions)
