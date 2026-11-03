"""SAM 3.1 self-hosted on Modal, as a drop-in for the Model API's ``sam-3.1``.

The ``kappa-sam`` Modal app (``tracker/app.py`` in this repository) runs Meta's
open-weight SAM 3.1 (``sam3.1_multiplex.pt``) on a cloud GPU and streams back the same
lines the hosted API streams. ``ModalSam`` has what :mod:`kappa.sam.client` needs from
a backend, ``store``, ``upload(path) -> file_id`` and ``stream_text(body, on_delta)``,
so ``track_concept`` and ``track_prompts`` run unchanged on either one.

Selected with ``KAPPA_SAM_BACKEND=modal`` (the default stays ``api``). Raw outputs are
cached in the content-addressed store under keys that name this backend, the
checkpoint and the service version, so they never mix with Model API outputs.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kappa.sam.client import ModelApiError
from kappa.store import Store
from kappa.video import sha256_file

APP_NAME = "kappa-sam"
CLASS_NAME = "Sam31"
VOLUME_NAME = "kappa-sam-videos"
STATS_DICT = "kappa-sam-stats"
# What the service runs; part of every cache key. SERVICE_VERSION must equal the one in
# tracker/app.py: it is sent with each call and a container of another version refuses it.
MODEL = "sam3.1_multiplex@0567debeec80"  # official checkpoint, sha256 prefix
SAM3_COMMIT = "f1647e8d0c1d"
SERVICE_VERSION = 3  # 3: 400-frame chunks, 48-frame overlap, ids carried by re-prompting
FILE_PREFIX = "modal:"


class ModalSamError(ModelApiError):
    pass


class ModalSam:
    """The self-hosted segmenter. ``service``/``volume``/``stats`` are injectable for tests."""

    backend = "modal"

    def __init__(
        self,
        store: Store | None = None,
        *,
        gpu: str | None = None,
        chunk_frames: int | None = None,
        overlap: int | None = None,
        carry: str | None = None,
        service: Any = None,
        volume: Any = None,
        stats: Any = None,
    ) -> None:
        self.store = store
        self.gpu = gpu or os.environ.get("KAPPA_SAM_MODAL_GPU") or None
        self.options = {
            k: v for k, v in (("chunk_frames", chunk_frames), ("overlap", overlap), ("carry", carry)) if v is not None
        }
        self._service, self._volume, self._stats = service, volume, stats
        self.last_stats: dict[str, Any] | None = None

    @property
    def model(self) -> str:
        return f"{MODEL} (self-hosted, sam3@{SAM3_COMMIT}, kappa-sam v{SERVICE_VERSION})"

    @property
    def cache_prefix(self) -> str:
        options = json.dumps(self.options, sort_keys=True)
        return f"sam:modal:{MODEL}:sam3@{SAM3_COMMIT}:v{SERVICE_VERSION}:{options}:"

    # -- modal handles (lazy: importing this module needs no modal) ------------

    def _modal(self):
        try:
            import modal
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise ModalSamError("the modal backend needs `pip install kappa[modal]`") from error
        return modal

    def service(self):
        if self._service is None:
            cls = self._modal().Cls.from_name(APP_NAME, CLASS_NAME)
            if self.gpu:
                cls = cls.with_options(gpu=self.gpu)
            self._service = cls()
        return self._service

    def volume(self):
        if self._volume is None:
            self._volume = self._modal().Volume.from_name(VOLUME_NAME, create_if_missing=True)
        return self._volume

    def stats_store(self):
        if self._stats is None:
            self._stats = self._modal().Dict.from_name(STATS_DICT, create_if_missing=True)
        return self._stats

    # -- backend protocol --------------------------------------------------------

    def close(self) -> None:
        pass

    def upload(self, path: Path) -> str:
        """Put the video in the service's volume once, keyed by its content."""
        digest = sha256_file(path)
        key = f"modal-file:{digest}"
        if self.store is not None and (known := self.store.recall(key)):
            return known
        remote = f"videos/{digest}.mp4"
        try:
            with self.volume().batch_upload(force=True) as batch:
                batch.put_file(str(path), remote)
        except Exception as error:
            raise ModalSamError(f"upload to the {VOLUME_NAME} volume failed: {error}") from error
        file_id = FILE_PREFIX + remote
        if self.store is not None:
            self.store.remember(key, file_id)
        return file_id

    def stream_text(self, body: dict[str, Any], on_delta: Callable[[str], None] | None = None) -> str:
        """Run a Model API-shaped SAM request on the service; return the concatenated lines.

        Raises rather than returning partial output, like the API client.
        """
        text, file_id = _request_parts(body)
        if not file_id.startswith(FILE_PREFIX):
            raise ModalSamError(f"{file_id!r} was not uploaded through the modal backend")
        direction = body.get("metadata", {}).get("propagation_direction", "both")
        job = uuid.uuid4().hex
        parts: list[str] = []
        try:
            for line in self.service().track.remote_gen(
                file_id[len(FILE_PREFIX):], text, direction, job=job, service_version=SERVICE_VERSION,
                **self.options,
            ):
                parts.append(line)
                if on_delta is not None:
                    on_delta(line)
        except ModelApiError:
            raise
        except Exception as error:
            raise ModalSamError(f"kappa-sam failed after {len(parts)} lines: {error}") from error
        try:
            self.last_stats = self.stats_store().get(job)
        except Exception:  # noqa: BLE001 - statistics are best effort
            self.last_stats = None
        return "".join(parts)


def _request_parts(body: dict[str, Any]) -> tuple[str, str]:
    """The prompt text and the video's file id from a Responses request body."""
    try:
        content = body["input"][0]["content"]
        text = next(c["text"] for c in content if c.get("type") == "input_text")
        file_id = next(c["file_id"] for c in content if c.get("type") == "input_video")
    except (KeyError, IndexError, StopIteration, TypeError) as error:
        raise ModalSamError("the request has no input_text and input_video file_id") from error
    return text, file_id
