"""The self-hosted SAM 3.1 backend against a fake Modal service (no network, no GPU)."""

from contextlib import contextmanager

import numpy as np
import pytest
from meta_sam_parser import FramePrompt, PromptBox, PromptObject
from test_pipeline import render
from test_runner import FakeApi, runner  # noqa: F401 - the runner fixture

from kappa.repair import Anchor
from kappa.sam.client import ModelApi, ModelApiError, track_concept, track_prompts
from kappa.sam.modal_backend import ModalSam, ModalSamError
from kappa.store import Store

LINES = ["<0f>0<|box;...|>\n", "<1f>0<|box;...|>\n"]


class FakeTrack:
    def __init__(self, lines, fail_after=None):
        self.lines, self.fail_after, self.calls = lines, fail_after, []

    def remote_gen(self, video, text, direction, **kwargs):
        self.calls.append((video, text, direction, kwargs))
        for i, line in enumerate(self.lines):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("container died")
            yield line


class FakeService:
    def __init__(self, lines=LINES, fail_after=None):
        self.track = FakeTrack(lines, fail_after)


class FakeVolume:
    def __init__(self):
        self.files = []

    @contextmanager
    def batch_upload(self, force=False):
        class Batch:
            def put_file(_, local, remote):
                self.files.append((local, remote))

        yield Batch()


class FakeStats:
    def get(self, job):
        return {"job": job, "fps": 12.5}


def _sam(tmp_path, **kw):
    return ModalSam(store=Store(tmp_path / "store"), service=FakeService(**kw), volume=FakeVolume(), stats=FakeStats())


def test_concept_streams_lines_and_caches_them(tmp_path):
    sam = _sam(tmp_path)
    seen = []
    raw, digest = track_concept(sam, "modal:videos/abc.mp4", "mouse", on_delta=seen.append)
    assert raw == "".join(LINES) and seen == LINES
    (video, text, direction, kwargs), = sam.service().track.calls
    assert (video, text, direction) == ("videos/abc.mp4", "mouse", "both")
    assert set(kwargs) == {"job", "service_version"} and sam.last_stats["fps"] == 12.5
    assert track_concept(sam, "modal:videos/abc.mp4", "mouse") == (raw, digest)
    assert len(sam.service().track.calls) == 1  # served from the store


def test_cache_keys_never_mix_backends_or_service_options(tmp_path):
    store = Store(tmp_path / "store")
    sam = ModalSam(store=store, service=FakeService(), volume=FakeVolume(), stats=FakeStats())
    track_concept(sam, "modal:videos/abc.mp4", "mouse")

    class Api(ModelApi):
        def __init__(self):
            self.store, self.streams = store, 0

        def stream_text(self, body, on_delta=None):
            self.streams += 1
            return "api output"

    api = Api()
    assert track_concept(api, "modal:videos/abc.mp4", "mouse")[0] == "api output"
    assert api.streams == 1
    other = ModalSam(store=store, chunk_frames=300, service=FakeService(["x\n"]), volume=FakeVolume(), stats=FakeStats())
    assert track_concept(other, "modal:videos/abc.mp4", "mouse")[0] == "x\n"
    assert other.service().track.calls[0][3]["chunk_frames"] == 300


def test_prompts_carry_grammar_and_direction(tmp_path):
    sam = _sam(tmp_path)
    obj = PromptObject(id=3, prompts=[FramePrompt(frame=40, box=PromptBox(10, 20, 30, 40), points=())])
    track_prompts(sam, "modal:videos/abc.mp4", (928, 576), [obj], "forward")
    (_, text, direction, _), = sam.service().track.calls
    assert text.startswith("<40f>3<|box;") and direction == "forward"


def test_upload_is_content_addressed_and_remembered(tmp_path):
    sam = _sam(tmp_path)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    first = sam.upload(clip)
    assert first.startswith("modal:videos/") and first.endswith(".mp4")
    assert sam.upload(clip) == first
    assert len(sam.volume().files) == 1


def test_a_failure_mid_stream_raises_instead_of_returning_partial_output(tmp_path):
    sam = _sam(tmp_path, fail_after=1)
    with pytest.raises(ModalSamError) as info:
        track_concept(sam, "modal:videos/abc.mp4", "mouse")
    assert isinstance(info.value, ModelApiError) and "after 1 lines" in str(info.value)
    assert sam.store.recall(sam.cache_prefix) is None


def test_foreign_file_ids_are_refused(tmp_path):
    with pytest.raises(ModalSamError):
        track_concept(_sam(tmp_path), "file-123", "mouse")


def test_runner_tracks_and_repairs_on_the_selected_backend(runner, monkeypatch):  # noqa: F811
    r, src = runner
    path = np.column_stack([np.linspace(100, 700, 150), np.full(150, 400.0)])
    raw = render({0: path}, range(150))
    r._sam = ModalSam(store=r.store, service=FakeService(raw.splitlines(keepends=True)), volume=FakeVolume(),
                      stats=FakeStats())
    monkeypatch.setenv("KAPPA_SAM_BACKEND", "modal")
    r.create(src, run_id="demo")
    state = r.draft("demo", "How far does the mouse walk?")
    r.confirm("demo", state.draft, by="Dr. Example")
    state = r.track("demo")
    assert r._api.streams == 0  # the Model API was not asked to segment
    assert state.file_id is None and state.segmenter_model.startswith("sam3.1_multiplex")
    assert r.store.get_text(state.sam_main_sha256) == raw
    r.repair("demo", [Anchor(0, 40, point=(round(path[40, 0]), 400))], by="A. Person")
    calls = r._sam.service().track.calls
    assert len(calls) == 2 and calls[1][1].startswith("<40f>0<|point;") and calls[1][2] == "forward"
    assert len(r._sam.volume().files) == 1  # uploaded once
    assert r.manifest("demo")["models"]["segmenter"].startswith("sam3.1_multiplex")


def test_unknown_backend_is_an_error(runner, monkeypatch):  # noqa: F811
    from kappa.runner import RunError

    r, _ = runner
    monkeypatch.setenv("KAPPA_SAM_BACKEND", "gpu-in-my-closet")
    with pytest.raises(RunError):
        _ = r.sam
    monkeypatch.setenv("KAPPA_SAM_BACKEND", "api")
    assert r.sam is r.api
