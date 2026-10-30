import json

import httpx
import pytest
from meta_sam_parser import FramePrompt, PromptBox, PromptObject

from kappa.sam.client import ModelApi, ModelApiError, track_concept, track_prompts
from kappa.store import Store


def _sse(events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


class Recorder:
    def __init__(self, events, status=200):
        self.events, self.status, self.requests = events, status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": "file-123", "object": "file"})
        return httpx.Response(self.status, text=_sse(self.events), headers={"content-type": "text/event-stream"})


def _api(recorder, tmp_path):
    return ModelApi(api_key="k", store=Store(tmp_path / "store"), transport=httpx.MockTransport(recorder))


DELTAS = [
    {"type": "response.created"},
    {"type": "response.output_text.delta", "delta": "<0f>0<|box;...|>\n"},
    {"type": "response.output_text.delta", "delta": "<1f>0<|box;...|>\n"},
    {"type": "response.completed"},
]


def test_stream_concatenates_deltas_and_reports_progress(tmp_path):
    rec = Recorder(DELTAS)
    seen = []
    raw, digest = track_concept(_api(rec, tmp_path), "file-123", "mouse", on_delta=seen.append)
    assert raw == "<0f>0<|box;...|>\n<1f>0<|box;...|>\n"
    assert seen == ["<0f>0<|box;...|>\n", "<1f>0<|box;...|>\n"]
    body = json.loads(rec.requests[0].content)
    assert body["model"] == "sam-3.1" and body["stream"] is True
    assert body["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}
    assert body["input"][0]["content"][0] == {"type": "input_text", "text": "mouse"}
    assert len(digest) == 64


def test_identical_request_is_served_from_store(tmp_path):
    rec = Recorder(DELTAS)
    api = _api(rec, tmp_path)
    first = track_concept(api, "file-123", "mouse")
    second = track_concept(api, "file-123", "mouse")
    assert first == second
    assert len(rec.requests) == 1


@pytest.mark.parametrize("kind", ["response.failed", "response.incomplete", "error"])
def test_failed_stream_raises_instead_of_returning_partial_output(tmp_path, kind):
    rec = Recorder([DELTAS[1], {"type": kind}])
    with pytest.raises(ModelApiError):
        track_concept(_api(rec, tmp_path), "file-123", "mouse")


def test_http_error_raises(tmp_path):
    rec = Recorder([], status=400)
    with pytest.raises(ModelApiError):
        track_concept(_api(rec, tmp_path), "file-123", "mouse")


def test_upload_is_cached_by_content(tmp_path):
    rec = Recorder(DELTAS)
    api = _api(rec, tmp_path)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    assert api.upload(clip) == "file-123"
    assert api.upload(clip) == "file-123"
    assert sum(r.url.path.endswith("/files") for r in rec.requests) == 1


def test_prompt_request_carries_subject_ids_and_direction(tmp_path):
    rec = Recorder(DELTAS)
    obj = PromptObject(id=2, prompts=[FramePrompt(frame=120, box=PromptBox(10, 20, 60, 50))])
    track_prompts(_api(rec, tmp_path), "file-123", (928, 576), [obj], direction="both")
    body = json.loads(rec.requests[0].content)
    text = body["input"][0]["content"][0]["text"]
    assert text.startswith("<120f>2<|box;")
    assert body["metadata"]["propagation_direction"] == "both"


def test_missing_key_is_an_error(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ModelApiError):
        ModelApi()
