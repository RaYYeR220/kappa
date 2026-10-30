"""A fake Model API and synthetic scenes for the CLI and MCP tests (no key, no network)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sam_synth import ellipse_mask, line, record
from synth_detections import touch_and_turn_back
from test_runner import CORNERS, DRAFT, REVIEW, _resp

from kappa.pipeline import analyse
from kappa.protocol import AssayProtocol
from kappa.video import VideoInfo

FPS, N, SIZE = 30.0, 150, 800
VIDEO = VideoInfo(width=SIZE, height=SIZE, fps=FPS, n_frames=N, duration_s=N / FPS, codec="h264")
MASK = ellipse_mask(20, 40)
VERDICT = {"present": True, "confidence": 0.9, "onset_s": 0.5, "offset_s": 1.2, "rationale": "Stands on its hind legs."}


def make_video(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={SIZE}x{SIZE}:rate={FPS:g}:duration=5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def render(paths: dict[int, np.ndarray], frames: range, mask_for: Callable[[int], np.ndarray] = lambda f: MASK) -> str:
    out = []
    for f in frames:
        mask = mask_for(f)
        h, w = mask.shape
        recs = [
            record(obj, round(p[f, 0] - w / 2), round(p[f, 1] - h / 2), mask, SIZE, SIZE, 0.95)
            for obj, p in paths.items()
            if not np.isnan(p[f, 0])
        ]
        if recs:
            out.append(line(f, recs))
    return "\n".join(out) + "\n"


def protocol(count: int = 1, rearing: bool = False) -> dict:
    metrics = [
        {"id": "distance", "kind": "distance"},
        {"id": "center_time", "kind": "time_in_zone", "zone": "center"},
    ]
    behaviours = []
    if rearing:
        metrics.append({"id": "rearing_count", "kind": "behaviour_count", "behaviour": "rearing"})
        behaviours.append({
            "id": "rearing", "label": "Rearing", "detector": "rearing_topdown",
            "definition": "The mouse stands on its hind legs with both forepaws off the floor.",
        })
    return AssayProtocol.model_validate({
        "title": "t",
        "question": "q",
        "subject": {"concept": "mouse", "expected_count": count},
        "arena": {"shape": "rect", "width_cm": 80, "height_cm": 80, "corners_px": [[0, 0], [800, 0], [800, 800], [0, 800]]},
        "zones": [{"id": "center", "kind": "center_fraction", "fraction": 0.25}],
        "metrics": metrics,
        "behaviours": behaviours,
        "smoothing": {"method": "none"},
        # the touch-and-turn-back fixture pauses ~0.9 body lengths apart
        "integrity": {"contact_distance": 1.0},
    }).model_dump(mode="json")


def walk() -> np.ndarray:
    return np.column_stack([np.linspace(100, 700, N), np.full(N, 400.0)])


def one_mouse(rearing: bool = False) -> str:
    """A mouse walking 60 cm in a straight line; with ``rearing`` its outline shrinks mid-walk."""
    small = ellipse_mask(14, 24)
    return render({0: walk()}, range(N), (lambda f: small if 60 <= f < 80 else MASK) if rearing else (lambda f: MASK))


def two_mice(rearing: bool = False) -> tuple[str, int, str, tuple[np.ndarray, np.ndarray]]:
    """Two mice that touch and turn back: (raw, fix frame, re-track for that frame, paths).

    With ``rearing`` both outlines shrink early on, long before they meet, so each
    mouse has a rearing candidate whose count the later contact locks.
    """
    a, b = touch_and_turn_back(N)
    small = ellipse_mask(14, 24)
    raw = render({7: a, 9: b}, range(N), (lambda f: small if 20 <= f < 40 else MASK) if rearing else (lambda f: MASK))
    analysis = analyse(AssayProtocol.model_validate(protocol(count=2)), VIDEO, raw, workers=1)
    fix = next(e.fix_frame for e in analysis.events if e.is_open_blocker and e.fix_frame is not None)
    return raw, fix, render({1: a, 2: b}, range(fix, N)), (a, b)


class FakeApi:
    """Answers each kind of request the runner makes; streams segmenter output line by line."""

    def __init__(self, store, main: str, retrack: str = "") -> None:
        self.store, self.main, self.retrack = store, main, retrack
        self.replies = {"assay_protocol": DRAFT, "points": CORNERS, "review": REVIEW, "verdict": VERDICT}
        self.calls: list[str] = []

    def upload(self, path):
        return "file-1"

    def create(self, body):
        name = body["text"]["format"]["name"]
        self.calls.append(name)
        return _resp(self.replies[name])

    def stream_text(self, body, on_delta=None):
        prompt = body["input"][0]["content"][0]["text"]
        raw = self.main if prompt == "mouse" else self.retrack
        self.calls.append("track" if prompt == "mouse" else "retrack")
        if on_delta:
            for chunk in raw.splitlines(keepends=True):
                on_delta(chunk)
        return raw


def write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def with_pin(argv: list[str], runner) -> list[str]:
    """What a careful person types: the sha256 of the protocol or marks they just reviewed.

    ``--unpinned`` in argv leaves the hash out, to test the refusal.
    """
    import json as _json

    from kappa import public as _public

    argv = [str(a) for a in argv]
    if "--unpinned" in argv:
        return [a for a in argv if a != "--unpinned"]
    if "--sha256" in argv or len(argv) < 2:
        return argv
    if argv[0] == "confirm":
        if "--protocol" in argv:
            try:
                body = _json.loads(Path(argv[argv.index("--protocol") + 1]).read_text(encoding="utf-8"))
            except ValueError:
                return argv
        else:
            try:
                body = _public.read_draft(runner, runner.load(argv[1]))
            except Exception:  # noqa: BLE001 - let the CLI report the real problem
                return argv
        try:
            return [*argv, "--sha256", _public.protocol_hash(body)]
        except Exception:  # noqa: BLE001 - an invalid protocol: the CLI reports it
            return argv
    if argv[0] == "repair" and "--from-proposal" in argv:
        frame = int(argv[argv.index("--from-proposal") + 1])
        try:
            return [*argv, "--sha256", _public.marks_hash(runner.repair_proposal(argv[1], frame))]
        except Exception:  # noqa: BLE001
            return argv
    return argv
