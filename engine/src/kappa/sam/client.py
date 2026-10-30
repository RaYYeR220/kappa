"""Meta Model API client for the calls Kappa makes: file upload and streamed Responses.

Streaming is used for every segmentation request: frames arrive one line at a time,
so progress can be shown while a ten-minute clip is still being tracked, and long
jobs don't hit the non-streaming time limit.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
from meta_sam_parser import PromptObject, build_object_prompt

from kappa.store import Store
from kappa.video import sha256_file

BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"


class ModelApiError(RuntimeError):
    pass


def _sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    data: list[str] = []
    for line in lines:
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif line == "" and data:
            payload = "\n".join(data)
            data = []
            if payload == "[DONE]":
                return
            yield json.loads(payload)
    if data and data != ["[DONE]"]:
        yield json.loads("\n".join(data))


class ModelApi:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        store: Store | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 900.0,
    ) -> None:
        key = api_key or os.environ.get("MODEL_API_KEY")
        if not key:
            raise ModelApiError("MODEL_API_KEY is not set")
        self.store = store
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def upload(self, path: Path) -> str:
        """Upload a file once; later calls for the same content reuse the file id."""
        digest = sha256_file(path)
        if self.store is not None and (known := self.store.recall(f"file:{digest}")):
            return known
        with path.open("rb") as handle:
            response = self._http.post(
                "/files", data={"purpose": "user_data"}, files={"file": (path.name, handle)}
            )
        if response.status_code != 200:
            raise ModelApiError(f"upload failed ({response.status_code}): {response.text[:500]}")
        file_id = response.json()["id"]
        if self.store is not None:
            self.store.remember(f"file:{digest}", file_id)
        return file_id

    def stream_text(self, body: dict[str, Any], on_delta: Callable[[str], None] | None = None) -> str:
        """POST /responses with streaming; return the concatenated output text.

        Raises on an error or failed/incomplete response rather than returning a
        partial result that could be mistaken for a complete one.
        """
        parts: list[str] = []
        with self._http.stream(
            "POST", "/responses", json={**body, "stream": True}, headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                response.read()
                raise ModelApiError(f"request failed ({response.status_code}): {response.text[:500]}")
            for event in _sse_events(response.iter_lines()):
                kind = event.get("type", "")
                if kind == "response.output_text.delta":
                    delta = event.get("delta", "")
                    parts.append(delta)
                    if on_delta is not None:
                        on_delta(delta)
                elif kind in {"response.failed", "response.incomplete", "error"}:
                    raise ModelApiError(f"{kind}: {json.dumps(event)[:500]}")
        return "".join(parts)

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self._http.post("/responses", json=body)
        if response.status_code != 200:
            raise ModelApiError(f"request failed ({response.status_code}): {response.text[:500]}")
        return response.json()


def _video_body(file_id: str, text: str, metadata: dict[str, str]) -> dict[str, Any]:
    return {
        "model": SAM_MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_video", "file_id": file_id},
                ],
            }
        ],
        "metadata": metadata,
    }


def _cached_stream(api: ModelApi, body: dict[str, Any], on_delta) -> tuple[str, str]:
    key = "sam:" + json.dumps(body, sort_keys=True)
    store = api.store
    if store is not None and (digest := store.recall(key)):
        raw = store.get_text(digest)
        if on_delta is not None:
            on_delta(raw)
        return raw, digest
    raw = api.stream_text(body, on_delta)
    digest = store.put_text(raw) if store is not None else ""
    if store is not None:
        store.remember(key, digest)
    return raw, digest


def track_concept(
    api: ModelApi, file_id: str, concept: str, on_delta: Callable[[str], None] | None = None
) -> tuple[str, str]:
    """Track every instance of a noun-phrase concept through a video. Returns (raw, sha256)."""
    body = _video_body(file_id, concept, {"include_confidence": "true", "mask_encoding": "one_bit"})
    return _cached_stream(api, body, on_delta)


def track_prompts(
    api: ModelApi,
    file_id: str,
    size: tuple[int, int],
    objects: Sequence[PromptObject],
    direction: str = "both",
    on_delta: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Track objects a person marked with boxes/points on chosen frames (identity repair).

    ``size`` is the video's (width, height); object ids are the subject numbers, so the
    returned tracks come back already labelled with the identities the person chose.
    """
    text = build_object_prompt(size=size, objects=list(objects))
    body = _video_body(
        file_id, text, {"include_confidence": "true", "propagation_direction": direction}
    )
    return _cached_stream(api, body, on_delta)
