import shutil
import subprocess

import pytest

from kappa.video import extract_clip, extract_frame, normalise, probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("src") / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def test_probe_counts_frames(source):
    info = probe(source)
    assert (info.width, info.height, info.n_frames) == (320, 240, 120)
    assert info.fps == pytest.approx(30)


def test_normalise_trims_resamples_and_names_by_hash(source, tmp_path):
    out, info, digest = normalise(source, tmp_path, start_s=1.0, end_s=3.0, fps=25)
    assert out.name == f"{digest}.mp4"
    assert info.fps == pytest.approx(25)
    assert info.n_frames == 50
    again, _, digest2 = normalise(source, tmp_path / "b", start_s=1.0, end_s=3.0, fps=25)
    assert digest2 == digest


def test_frame_and_clip_extraction(source, tmp_path):
    frame = extract_frame(source, 30, 30.0, tmp_path / "f.jpg")
    assert frame.stat().st_size > 0
    clip = extract_clip(source, 30, 60, 30.0, tmp_path / "c.mp4")
    assert probe(clip).n_frames == 30
