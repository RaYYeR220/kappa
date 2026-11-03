import json

import httpx
import pytest

from kappa.agents.protocol_agent import output_text
from kappa.sam.client import ModelApiError
from kappa.venice import VeniceApi, choose_model, to_chat

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def body(model="muse-spark-1.3", video=False):
    content = [{"type": "input_text", "text": "q"}, {"type": "input_image", "image_url": "data:image/jpeg;base64,AA=="}]
    if video:
        content.append({"type": "input_video", "video_url": "data:video/mp4;base64,AA=="})
    return {
        "model": model,
        "instructions": "be strict",
        "input": [{"type": "message", "role": "user", "content": content}],
        "reasoning": {"effort": "max"},
        "text": {"format": {"type": "json_schema", "name": "r", "schema": SCHEMA}},
        "max_output_tokens": 900,
    }


def test_model_mapping():
    assert choose_model("muse-spark-1.3", with_video=False) == "muse-spark-open"
    assert choose_model("muse-spark-1.3", with_video=True) == "muse-wonder"
    assert choose_model("muse-wonder-1.0", with_video=False) == "muse-wonder"
    with pytest.raises(ModelApiError):
        choose_model("sam-3.1", with_video=True)


def test_request_translation():
    chat = to_chat(body(video=True))
    assert chat["model"] == "muse-wonder" and chat["reasoning_effort"] == "max" and chat["max_tokens"] == 900
    assert chat["messages"][0] == {"role": "system", "content": "be strict"}
    parts = chat["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "video_url"]
    assert chat["response_format"]["json_schema"]["schema"] == SCHEMA
    assert to_chat(body())["reasoning_effort"] == "xhigh"


def test_create_returns_a_responses_shaped_result():
    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        assert sent["model"] == "muse-spark-open"
        return httpx.Response(200, json={"model": "muse-spark-open", "choices": [{"finish_reason": "stop", "message": {"content": '```json\n{"a": 3}\n```'}}]})

    api = VeniceApi(api_key="k", transport=httpx.MockTransport(handler))
    result = api.create(body())
    assert json.loads(output_text(result)) == {"a": 3} and result["model"] == "muse-spark-open"


@pytest.mark.parametrize("status,payload", [(500, {"error": "x"}), (200, {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]})])
def test_failures_raise(status, payload):
    api = VeniceApi(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(status, json=payload)))
    with pytest.raises(ModelApiError):
        api.create(body())
