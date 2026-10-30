import pytest
from pydantic import ValidationError

from kappa.protocol import AssayProtocol


def open_field(**overrides):
    base = dict(
        title="Open field",
        question="How far does the mouse travel and how long does it spend in the centre?",
        subject={"concept": "mouse", "expected_count": 1},
        arena={"shape": "rect", "width_cm": 45, "height_cm": 45},
        zones=[{"id": "center", "kind": "center_fraction", "fraction": 0.25}],
        metrics=[
            {"id": "distance", "kind": "distance"},
            {"id": "center_time", "kind": "time_in_zone", "zone": "center"},
        ],
    )
    base.update(overrides)
    return AssayProtocol.model_validate(base)


def test_valid_protocol_hash_is_stable_and_content_addressed():
    a, b = open_field(), open_field()
    assert a.content_hash() == b.content_hash()
    c = open_field(zones=[{"id": "center", "kind": "center_fraction", "fraction": 0.5}])
    assert c.content_hash() != a.content_hash()


@pytest.mark.parametrize(
    "overrides",
    [
        {"metrics": [{"id": "x", "kind": "time_in_zone", "zone": "nowhere"}]},
        {"metrics": [{"id": "x", "kind": "time_in_zone"}]},
        {"metrics": [{"id": "x", "kind": "behaviour_count"}]},
        {"metrics": [{"id": "x", "kind": "distance"}, {"id": "x", "kind": "mean_speed"}]},
        {"metrics": [{"id": "x", "kind": "mean_inter_subject_distance"}]},
        {"metrics": []},
        {"zones": [{"id": "center", "kind": "center_fraction"}]},
        {"arena": {"shape": "circle", "width_cm": 40, "height_cm": 45}},
        {"arena": {"shape": "rect", "width_cm": 45, "height_cm": 45, "corners_px": [[0, 0], [1, 0], [1, 1]]}},
        {"subject": {"concept": "mouse", "expected_count": 17}},
        {"unexpected": True},
    ],
)
def test_invalid_protocols_are_rejected(overrides):
    with pytest.raises(ValidationError):
        open_field(**overrides)


def test_protocol_is_immutable():
    p = open_field()
    with pytest.raises(ValidationError):
        p.title = "changed"
