import json

import pytest

from kappa.agents.adjudicator import Candidate, adjudicate, judge
from kappa.model import Span
from kappa.protocol import BehaviourSpec

REAR = BehaviourSpec(
    id="rear",
    label="Rearing",
    definition="The animal raises its forepaws off the floor and stands on its hind legs.",
    detector="rearing_topdown",
)


def _resp(obj):
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(obj)}]}]}


class FakeApi:
    def __init__(self, reply=None, fail=False):
        self.reply, self.fail, self.bodies = reply, fail, []

    def create(self, body):
        self.bodies.append(body)
        if self.fail:
            raise RuntimeError("boom")
        return _resp(self.reply)


@pytest.fixture
def cand(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return Candidate(subject=1, span=Span(1000, 1100), clip=clip, clip_start=990)


def test_present_verdict_maps_clip_seconds_to_frames(cand):
    api = FakeApi({"present": True, "confidence": 0.9, "onset_s": 1.0, "offset_s": 2.2, "rationale": "stands up"})
    v = judge(api, REAR, cand, fps=25)
    assert v.present and v.span == Span(1015, 1045)
    body = api.bodies[0]
    assert body["input"][0]["content"][1]["video_url"].startswith("data:video/mp4;base64,")
    assert "Operational definition" in body["input"][0]["content"][0]["text"]


def test_span_is_clamped_to_candidate(cand):
    api = FakeApi({"present": True, "confidence": 0.9, "onset_s": 0.0, "offset_s": 99.0, "rationale": ""})
    assert judge(api, REAR, cand, fps=25).span == Span(1000, 1100)


def test_failed_call_is_an_error_not_a_no(cand):
    v = judge(FakeApi(fail=True), REAR, cand, fps=25)
    assert v.error is not None and not v.present


def test_adjudicate_keeps_confident_present_verdicts(cand):
    yes = FakeApi({"present": True, "confidence": 0.8, "onset_s": 1.0, "offset_s": 2.0, "rationale": "r"})
    events, verdicts = adjudicate(yes, REAR, [cand, cand], fps=25, workers=2)
    assert len(events) == 2 and events[0].behaviour == "rear" and events[0].source == "muse-spark-1.3"
    unsure = FakeApi({"present": True, "confidence": 0.3, "onset_s": 1.0, "offset_s": 2.0, "rationale": "r"})
    events, verdicts = adjudicate(unsure, REAR, [cand], fps=25)
    assert events == [] and len(verdicts) == 1
