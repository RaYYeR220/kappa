"""The service's wire format, checked against the engine's parser and Meta's reference parser.

Run with the engine's environment::

    cd engine && uv run pytest ../tracker/tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_wire import (
    PromptError,
    bf16,
    carry_prompt,
    format_confidence,
    format_line,
    is_object_prompt,
    mask_box,
    parse_object_prompts,
)

from kappa.sam.parse import parse_output


def _ellipse(h, w, cy, cx, ry, rx):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1


def test_lines_parse_with_engine_and_reference_parsers():
    w, h = 320, 200
    a = _ellipse(h, w, 50, 60, 20, 30)
    b = _ellipse(h, w, 120, 200, 15, 40)
    raw = "".join(
        line
        for f in range(3)
        if (line := format_line(f, np.array([7, 2]), np.stack([a, b]), np.array([0.96, 0.5]), w, h))
    )
    assert raw.count("\n") == 3 and raw.startswith("<0f>2<|box;")
    fast = parse_output(raw)
    strict = parse_output(raw, strict=True)
    assert [(d.frame, d.track_id, d.box, d.area_px) for d in fast.detections] == [
        (d.frame, d.track_id, d.box, d.area_px) for d in strict.detections
    ]
    assert not fast.diagnostics and (fast.width, fast.height) == (w, h)
    d2 = next(d for d in fast.detections if d.track_id == 2 and d.frame == 0)
    x1, y1, x2, y2 = mask_box(b)
    assert d2.box == (x1, y1, x2 + 1, y2 + 1)  # wire box inclusive, parser exclusive
    assert d2.area_px == b.sum()
    assert d2.confidence == 0.5


def test_empty_frames_and_masks_emit_nothing():
    w, h = 10, 10
    empty = np.zeros((h, w), dtype=bool)
    assert format_line(0, np.array([1]), empty[None], np.array([0.9]), w, h) is None
    assert format_line(0, np.array([], dtype=np.int64), np.zeros((0, h, w), bool), np.array([]), w, h) is None


def test_confidence_is_a_plain_decimal_in_bf16():
    assert format_confidence(0.9609375) == "0.9609375"
    assert format_confidence(1.0) == "1.0"
    assert format_confidence(1e-12) == "0.0"
    assert format_confidence(-3) == "0.0" and format_confidence(7) == "1.0"
    assert "e" not in format_confidence(3.3e-5)
    assert bf16(0.96) == 0.9609375


def test_object_prompts_parse_like_the_api_documents():
    text = (
        "<30f>1<|point;x=585;y=366;w=960;h=540|><75f>2<|box;x1=405;y1=337;x2=516;y2=420;w=960;h=540|>"
        "<120f>1<|box;x1=493;y1=320;x2=641;y2=444;w=960;h=540|>"
        "<160f>2<|point;x=533;y=403;w=960;h=540|>-<|point;x=482;y=405;w=960;h=540|>"
    )
    assert is_object_prompt(text) and not is_object_prompt("red bicycle")
    prompts = parse_object_prompts(text, 1920, 1080)  # coordinates scale to the media
    assert [(p.frame, p.obj_id) for p in prompts] == [(30, 1), (75, 2), (120, 1), (160, 2)]
    assert prompts[0].points == [(1170.0, 732.0, 1)]
    assert prompts[1].box == (810.0, 674.0, 1032.0, 840.0)
    assert prompts[3].points == [(1066.0, 806.0, 1), (964.0, 810.0, 0)]
    coords, labels = prompts[1].sam_inputs(1920, 1080)
    assert labels == [2, 3] and coords[0] == [810 / 1920, 674 / 1080] and coords[1] == [1033 / 1920, 841 / 1080]


def test_comma_separated_objects_on_one_frame():
    text = "<0f>1<|box;x1=444;y1=626;x2=675;y2=849;w=1800;h=1200|>,2<|point;x=1430;y=700;w=1800;h=1200|>-<|point;x=1300;y=450;w=1800;h=1200|>"
    p = parse_object_prompts(text, 1800, 1200)
    assert [(q.obj_id, q.box is not None, len(q.points)) for q in p] == [(1, True, 0), (2, False, 2)]


@pytest.mark.parametrize(
    "text",
    [
        "<0f>1<|box;x1=1;y1=1;x2=5;y2=5;w=10;h=10|>-<|box;x1=1;y1=1;x2=5;y2=5;w=10;h=10|>",
        "<0f>1-<|box;x1=1;y1=1;x2=5;y2=5;w=10;h=10|>",
        "<0f>1<|point;x=1;y=1;w=10;h=10|>,1<|point;x=2;y=2;w=10;h=10|>",
        "<0f>1<|point;x=1;y=1;w=10;h=10|>,2<|point;x=2;y=2;w=20;h=20|>",
        "<0f>1<|point;x=11;y=1;w=10;h=10|>",
        "<0f>1<|point;x=1;y=1;w=10;h=12|>",
        "<0f>1<|mask;x=0;y=0;data=1,1,!xxxxx|>",
        "<0f>1",
    ],
)
def test_invalid_prompts_are_rejected(text):
    with pytest.raises(PromptError):
        parse_object_prompts(text, 10, 10)


def test_more_than_sixteen_objects_is_rejected():
    text = "<0f>" + ",".join(f"{i}<|point;x=1;y=1;w=10;h=10|>" for i in range(17))
    with pytest.raises(PromptError):
        parse_object_prompts(text, 10, 10)


def test_carry_prompt_points_inside_the_body():
    # two touching ellipses in one mask: the box covers both, the point must sit in the bigger body
    m = _ellipse(100, 200, 50, 60, 30, 50) | _ellipse(100, 200, 50, 130, 10, 25)
    p = carry_prompt(5, 3, m)
    assert p.frame == 5 and p.obj_id == 3 and p.box == mask_box(m)
    (x, y, label), = p.points
    assert label == 1 and m[int(y), int(x)] and abs(x - 60) <= 3 and abs(y - 50) <= 3
    assert carry_prompt(0, 1, np.zeros((5, 5), bool)) is None


def test_match_ids_pairs_objects_by_overlap_and_leaves_newcomers_unmatched():
    from sam_wire import match_ids

    h, w = 60, 120
    left, right, far = _ellipse(h, w, 30, 25, 10, 15), _ellipse(h, w, 30, 90, 10, 15), _ellipse(h, w, 5, 60, 3, 3)
    previous = {p: (np.array([4, 9]), np.stack([left, right])) for p in range(10, 16)}
    # the new chunk calls them 1 (right) and 0 (left), plus a newcomer 2 nobody had
    pending = [(p, np.array([0, 1, 2]), np.stack([left, right, far])) for p in range(10, 16)]
    assert match_ids(previous, pending) == {0: 4, 1: 9}
    # an object present on only a few shared frames is not trusted
    sparse = [(p, np.array([0]), left[None]) for p in range(10, 11)]
    assert match_ids(previous, sparse + [(p, np.array([], dtype=np.int64), left[:0][None][:0]) for p in range(11, 16)]) == {}
    assert match_ids({}, pending) == {}


def test_match_ids_accepts_an_object_the_new_chunk_confirms_late():
    from sam_wire import match_ids

    h, w = 60, 120
    left, right = _ellipse(h, w, 30, 25, 10, 15), _ellipse(h, w, 30, 90, 10, 15)
    previous = {p: (np.array([4, 9]), np.stack([left, right])) for p in range(0, 16)}
    early = [(p, np.array([0]), left[None]) for p in range(0, 10)]
    late = [(p, np.array([0, 1]), np.stack([left, right])) for p in range(10, 16)]  # 1 shows up at frame 10
    assert match_ids(previous, early + late) == {0: 4, 1: 9}
