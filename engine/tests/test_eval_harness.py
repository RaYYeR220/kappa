import json
from pathlib import Path

from kappa.eval.harness import GRADED, TEST, TUNING, with_union
from kappa.eval.sturman import SturmanLabels
from kappa.protocol import AssayProtocol


def test_split_matches_the_preregistration():
    text = (Path(__file__).parents[2] / "eval" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    assert len(TUNING) == 5 and len(TEST) == 15 and not set(TUNING) & set(TEST)
    for video in TUNING + TEST:
        assert video in text


def test_union_of_supported_and_unsupported_rears():
    labels = SturmanLabels(
        bouts={"OFT_1": {"A": {"supported_rear": [(5.0, 6.0)], "unsupported_rear": [(1.0, 2.0)], "grooming": []}}},
        windows={"OFT_1": {"A": (0.0, 600.0)}},
        behaviours=("supported_rear", "unsupported_rear", "grooming"),
    )
    assert with_union(labels).bouts["OFT_1"]["A"]["rear"] == [(1.0, 2.0), (5.0, 6.0)]


def test_evaluation_protocol_is_valid_and_grades_every_behaviour():
    path = Path(__file__).parents[2] / "eval" / "protocols" / "sturman_open_field.json"
    protocol = AssayProtocol.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert {b.id for b in protocol.behaviours} == set(GRADED)
