"""Meta's Muse models through Venice's OpenAI-compatible API.

Kappa's model steps (protocol drafting, arena grounding, behaviour adjudication,
Reviewer 2) build Meta Model API Responses requests. ``VeniceApi.create`` accepts the
same request, sends it to Venice as a chat completion and returns a Responses-shaped
result, so those steps run unchanged on either provider.

Model mapping: requests for Muse Spark go to ``muse-spark-open`` (the open-weight
Muse Spark), except requests that carry video, which Muse Spark Open does not take:
those go to ``muse-wonder``. Requests for Muse Wonder go to ``muse-wonder``. The
model that actually answered is returned in ``model`` so records name it truthfully.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from kappa.sam.client import ModelApiError

BASE_URL = "https://api.venice.ai/api/v1"
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}
SPARK = "muse-spark-open"
WONDER = "muse-wonder"
_EFFORT = {
    SPARK: {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "xhigh"},
    WONDER: {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "max", "max": "max"},
}


def _part(part: dict[str, Any]) -> dict[str, Any]:
    kind = part.get("type")
    if kind == "input_text":
        return {"type": "text", "text": part["text"]}
    if kind == "input_image":
        url = part["image_url"]
        return {"type": "image_url", "image_url": {"url": url if isinstance(url, str) else url["url"]}}
    if kind == "input_video":
        if "video_url" not in part:
            raise ModelApiError("Venice takes videos inline or by URL, not by Files API id")
        return {"type": "video_url", "video_url": {"url": part["video_url"]}}
    raise ModelApiError(f"unsupported content part for Venice: {kind}")


def _has_video(body: dict[str, Any]) -> bool:
    return any(
        part.get("type") == "input_video"
        for item in body.get("input", [])
        for part in (item.get("content") if isinstance(item.get("content"), list) else [])
    )


def choose_model(requested: str, with_video: bool) -> str:
    if requested.startswith("muse-wonder"):
        return WONDER
    if requested.startswith("muse-spark"):
        return WONDER if with_video else SPARK
    raise ModelApiError(f"no Venice equivalent for {requested}")


def to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """A Model API Responses request as a Venice chat completion request."""
    model = choose_model(body["model"], _has_video(body))
    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    for item in body.get("input", []):
        content = item.get("content")
        parts = [_part(p) for p in content] if isinstance(content, list) else content
        messages.append({"role": item.get("role", "user"), "content": parts})
    chat: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "venice_parameters": {"include_venice_system_prompt": False, "strip_thinking_response": True},
    }
    fmt = (body.get("text") or {}).get("format")
    if fmt and fmt.get("type") == "json_schema":
        chat["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": fmt.get("name", "result"), "schema": fmt["schema"]},
        }
    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        chat["reasoning_effort"] = _EFFORT[model].get(effort, "high")
    if body.get("max_output_tokens"):
        chat["max_tokens"] = body["max_output_tokens"]
    return chat


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


class VeniceApi:
    """The ``create`` half of :class:`kappa.sam.client.ModelApi`, backed by Venice."""

    def __init__(self, api_key: str | None = None, transport: httpx.BaseTransport | None = None, timeout_s: float = 600.0) -> None:
        key = api_key or os.environ.get("VENICE_API_KEY")
        if not key:
            raise ModelApiError("VENICE_API_KEY is not set")
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def create(self, body: dict[str, Any], attempts: int = 6) -> dict[str, Any]:
        """One request; rate limits, server errors and timeouts are retried with backoff."""
        chat = to_chat(body)
        delay = 2.0
        for attempt in range(attempts):
            try:
                response = self._http.post("/chat/completions", json=chat)
            except httpx.TransportError as error:
                if attempt == attempts - 1:
                    raise ModelApiError(f"Venice request failed: {error}") from error
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if response.status_code in RETRYABLE and attempt < attempts - 1:
                wait = response.headers.get("retry-after")
                time.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else delay)
                delay = min(delay * 2, 60)
                continue
            break
        if response.status_code != 200:
            raise ModelApiError(f"Venice request failed ({response.status_code}): {response.text[:500]}")
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise ModelApiError("the model ran out of output tokens before finishing")
        text = _strip_fences((choice.get("message") or {}).get("content") or "")
        return {
            "model": data.get("model", chat["model"]),
            "provider": "venice",
            "usage": data.get("usage"),
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        }
