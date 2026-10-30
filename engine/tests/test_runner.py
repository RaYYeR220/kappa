"""The run lifecycle against a fake Model API: every step persists, nothing runs unconfirmed."""

import json
import shutil
import subprocess

import numpy as np
import pytest
from test_pipeline import render

from kappa.runner import Runner, RunError

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

DRAFT = {
    "title": "Open field",
    "subject_concept": "mouse",
    "subject_label": "mouse",
    "expected_count": 1,
    "arena_shape": "rect",
    "arena_width_cm": 80,
    "arena_height_cm": 80,
    "zones": [{"id": "center", "kind": "center_fraction", "fraction": 0.25, "band_cm": None, "rect_cm": None}],
    "entry_mode": "centroid",
    "metrics": [
        {"id": "distance", "kind": "distance", "zone": None, "behaviour": None, "bin_s": None},
        {"id": "center_time", "kind": "time_in_zone", "zone": "center", "behaviour": None, "bin_s": None},
    ],
    "behaviours": [],
    "max_speed_cm_s": 120,
    "refused": [],
    "assumptions": ["Arena is 80 x 80 cm."],
    "notes": "",
}
CORNERS = {"points": [
    {"object_name": n, "x": x, "y": y}
    for n, x, y in [
        ("the top-left corner of the arena floor", 0, 0),
        ("the top-right corner of the arena floor", 1000, 0),
        ("the bottom-right corner of the arena floor", 1000, 1000),
        ("the bottom-left corner of the arena floor", 0, 1000),
    ]
]}
REVIEW = {"verdict": "minor_concerns", "findings": [
    {"severity": "concern", "topic": "definition", "claim": "Centre zone of 25% is smaller than some labs use.", "refs": ["center_time"], "suggestion": "Report 50% too."}
]}


def _resp(obj):
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(obj)}]}]}


class FakeApi:
    def __init__(self, store, raw):
        self.store, self.raw = store, raw
        self.replies = [DRAFT, CORNERS, REVIEW]
        self.streams = 0

    def upload(self, path):
        return "file-1"

    def create(self, body):
        return _resp(self.replies.pop(0))

    def stream_text(self, body, on_delta=None):
        self.streams += 1
        if on_delta:
            on_delta(self.raw)
        return self.raw


@pytest.fixture
def runner(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=800x800:rate=30:duration=5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )
    r = Runner(home=tmp_path / "home")
    path = np.column_stack([np.linspace(100, 700, 150), np.full(150, 400.0)])
    r._api = FakeApi(r.store, render({0: path}, range(150)))
    return r, src


def test_full_lifecycle(runner):
    r, src = runner
    state = r.create(src, run_id="demo")
    assert state.video["n_frames"] == 150
    state = r.draft("demo", "How far does the mouse walk and how long is it in the centre?")
    assert state.draft["arena"]["corners_px"][2] == [799, 799]
    with pytest.raises(RunError):
        r.track("demo")
    with pytest.raises(RunError):
        r.confirm("demo", state.draft, by="  ")
    r.confirm("demo", state.draft, by="Dr. Example")
    r.track("demo")
    r.track("demo")
    assert r._api.streams == 1  # the second request is served from the store
    manifest = r.manifest("demo", review=True)
    distance = next(x for x in manifest["results"] if x["metric_id"] == "distance")
    assert distance["status"] == "verified" and distance["value"] == pytest.approx(60, rel=0.02)
    assert manifest["review"][0]["verdict"] == "minor_concerns"
    assert manifest["models"]["reviewer"] == "muse-wonder-1.0"
    assert (r.path("demo") / "manifest.json").exists()
    assert r.load("demo").manifest_sha256 == r.store.put_json(manifest)


def test_unknown_run(runner):
    r, _ = runner
    with pytest.raises(RunError):
        r.load("nope")
