"""Video probing and normalisation.

Everything downstream is keyed by the sha256 of the *normalised* file, so a frame
index means the same frame in the segmenter's output, in the metric engine and in
the evidence clips a person reviews.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_SEGMENTER_FRAMES = 15_000


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    codec: str

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "duration_s": self.duration_s,
            "codec": self.codec,
        }


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"{tool} is required but was not found on PATH")
    return path


def _ratio(text: str) -> float:
    num, _, den = text.partition("/")
    return float(num) / float(den or 1)


def probe(path: Path) -> VideoInfo:
    """Read stream properties; the frame count is counted, not estimated from duration."""
    out = subprocess.run(
        [
            _require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    fps = _ratio(stream["avg_frame_rate"])
    n_frames = int(stream["nb_read_frames"])
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        n_frames=n_frames,
        duration_s=float(stream.get("duration") or n_frames / fps),
        codec=stream["codec_name"],
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def normalise(
    src: Path,
    dest_dir: Path,
    start_s: float = 0.0,
    end_s: float | None = None,
    fps: float | None = None,
) -> tuple[Path, VideoInfo, str]:
    """Re-encode to constant-frame-rate H.264 with the index up front, optionally trimmed.

    Constant frame rate matters: the segmenter numbers frames, the metric engine turns
    frame counts into seconds, and a variable-rate file would make those disagree.
    Returns the normalised path (named by its hash), its properties and the hash.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    source = probe(src)
    rate = fps or source.fps
    tmp = dest_dir / f".normalising-{src.stem}.mp4"
    cmd = [_require("ffmpeg"), "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(src)]
    if end_s is not None:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    cmd += [
        "-an", "-vf", f"fps={rate}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    info = probe(tmp)
    if info.n_frames > MAX_SEGMENTER_FRAMES:
        tmp.unlink()
        raise ValueError(
            f"{info.n_frames} frames exceeds the segmenter's {MAX_SEGMENTER_FRAMES}-frame limit; "
            "trim the clip or lower the frame rate"
        )
    digest = sha256_file(tmp)
    final = dest_dir / f"{digest}.mp4"
    tmp.replace(final)
    return final, info, digest


def extract_frame(path: Path, frame: int, fps: float, dest: Path) -> Path:
    """Write one frame as JPEG (used for keyframes sent to the protocol agent)."""
    subprocess.run(
        [
            _require("ffmpeg"), "-y", "-v", "error", "-ss", f"{frame / fps:.4f}", "-i", str(path),
            "-frames:v", "1", "-q:v", "2", str(dest),
        ],
        check=True,
    )
    return dest


def extract_clip(path: Path, start_frame: int, end_frame: int, fps: float, dest: Path) -> Path:
    """Cut ``[start_frame, end_frame)`` into its own MP4 (evidence clips, adjudication)."""
    subprocess.run(
        [
            _require("ffmpeg"), "-y", "-v", "error", "-ss", f"{start_frame / fps:.4f}", "-i", str(path),
            "-frames:v", str(end_frame - start_frame), "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest),
        ],
        check=True,
    )
    return dest
