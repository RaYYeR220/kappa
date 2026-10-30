"""Public evaluation datasets: where they live, what they hash to, and a verified fetch.

Nothing here is redistributed. The Sturman et al. 2020 rater labels are GPL-3.0 (from
the DLCAnalyzer repository) and the videos CC-BY-4.0 (Zenodo), so the repository only
records their URLs and checksums; :func:`fetch` downloads them into a cache outside
the source tree on first use and refuses any file whose checksum does not match.

Sturman, O. et al. (2020) Deep learning-based behavioral analysis reaches human
accuracy and is capable of outperforming commercial solutions.
Neuropsychopharmacology 45:1942-1952. doi:10.1038/s41386-020-0776-y
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ZENODO_RECORD = "https://zenodo.org/records/3608658"  # doi:10.5281/zenodo.3608658
DLCANALYZER_RAW = "https://raw.githubusercontent.com/ETHZ-INS/DLCAnalyzer/master"

# 3D-ZeF (Pedersen et al. 2020, CVPR), the zebrafish identity benchmark. Not in SOURCES:
# the archive is 14.3 GB and no checksum has been pinned yet, and fetch only returns
# verified files. Pin it with sha256_file() after a first manual download.
ZEF_3D_HOMEPAGE = "https://motchallenge.net/data/3D-ZeF20/"
ZEF_3D_URL = "https://motchallenge.net/data/3DZeF20.zip"


@dataclass(frozen=True)
class Source:
    """One downloadable file. At least one checksum is required to fetch it.

    ``sha256`` is pinned from a local copy we inspected; ``md5`` is the checksum the
    host publishes (Zenodo), used where no local copy has been hashed yet. md5 guards
    against truncation and silent replacement, not against a malicious host.
    """

    url: str
    filename: str
    sha256: str | None = None
    md5: str | None = None
    size: int | None = None
    licence: str = ""
    note: str = ""


_GPL_LABELS = "GPL-3.0 (DLCAnalyzer); fetched at run time, never committed"
_CC_BY = "CC-BY-4.0 (Zenodo 3608658)"

# Open-field videos on Zenodo: (video id, size in bytes, Zenodo-published md5).
_OFT_VIDEOS = (
    ("OFT_5", 25799279, "622a2a8f39001bddfc02690b4f17f354"),
    ("OFT_6", 25214283, "5f887b9ea478c2d90c78cf32c55800f4"),
    ("OFT_11", 24917149, "cefa2b4b79290e7e0d8968bbdd31ef54"),
    ("OFT_12", 25433672, "7d3f2cc662eece1ad1e3d374c6c9d4a8"),
    ("OFT_14", 26188622, "b2f44a6af593e0b26cd5accbe8e2d2f0"),
    ("OFT_15", 25181458, "602e35b25662faf9038ede64b591f5c1"),
    ("OFT_16", 27388290, "8e6d99fa1dfba1c660b2bbccc6b91f4d"),
    ("OFT_23", 22285664, "fbdfe760a8528eb76dec331ed32bbd41"),
    ("OFT_24", 27075663, "a20298585e7ef46fd2f4b97019b9546b"),
    ("OFT_38", 24920912, "395d6b640ea5dda412175c7b4a9d422d"),
    ("OFT_39", 25461956, "10020247a7df86af91065d6a0d8311f6"),
    ("OFT_41", 25059492, "70e209d6b1ef3f5d3a9156e37e2fbca7"),
    ("OFT_43", 25979150, "06e46d3cd4a187c927edb73229416a0d"),
    ("OFT_44", 25032040, "b79d4f30df912dfb7f2dd79ce7560a8b"),
    ("OFT_49", 26622295, "abe01d92e761985a4e04dca62a85291a"),
    ("OFT_50", 26241782, "c9e6d25a873712efce963031c07179c3"),
    ("OFT_51", 26252409, "bf8c0f4e7340a2442b8b3a6ffd704e9e"),
    ("OFT_52", 23021985, "96e00e1e47ec1ec75548a6ea3267918c"),
    ("OFT_54", 21966187, "47220948545722e2bddb366310b7fc49"),
    ("OFT_58", 26225554, "53aa4791ef713bfd5f65e330550bf8a4"),
)
_OFT_VIDEO_SHA256 = {
    "OFT_23": "99da95f88711430b1335c16fae187562b4668310581e2baab713a2651e27b5bc",
}

SOURCES: Mapping[str, Source] = {
    "sturman_oft_labels": Source(
        url=f"{DLCANALYZER_RAW}/data/OFT/Labels/AllLabDataOFT_final.csv",
        filename="AllLabDataOFT_final.csv",
        sha256="2a690c586f00af4574be5555e63ff66698a3149ef8c70e050876f138e7fd5fd3",
        size=763726,
        licence=_GPL_LABELS,
        note="3 raters x 20 open-field videos: rears, grooming, jumping bouts; Start/End markers",
    ),
    "sturman_epm_labels": Source(
        url=f"{DLCANALYZER_RAW}/data/EPM/Labels/AllEPMLables.csv",
        filename="AllEPMLables.csv",
        sha256="46906df810429d2476fc2b3ac0aa079e98554054fc4851a55f2cd1c8939c3d2d",
        size=149802,
        licence=_GPL_LABELS,
        note="3 raters x 5 elevated-plus-maze videos",
    ),
    **{
        f"sturman_video_{vid}": Source(
            url=f"{ZENODO_RECORD}/files/{vid}.mp4?download=1",
            filename=f"{vid}.mp4",
            sha256=_OFT_VIDEO_SHA256.get(vid),
            md5=md5,
            size=size,
            licence=_CC_BY,
            note="open-field test, 25 fps",
        )
        for vid, size, md5 in _OFT_VIDEOS
    },
}


class DatasetIntegrityError(ValueError):
    """A downloaded or cached file does not match its pinned checksum or size."""


def default_cache_dir() -> Path:
    """``$KAPPA_DATA_DIR`` if set, else ``~/.cache/kappa/datasets`` — outside the repo, so
    copyleft data can never be committed by accident."""
    env = os.environ.get("KAPPA_DATA_DIR")
    return Path(env) if env else Path.home() / ".cache" / "kappa" / "datasets"


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file, for pinning a new source."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digests(path: Path) -> tuple[str, str, int]:
    sha, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def _mismatch(path: Path, src: Source) -> str | None:
    """Why ``path`` is not ``src``, or None if every pinned property matches."""
    sha, md5, size = _digests(path)
    if src.size is not None and size != src.size:
        return f"size {size} != pinned {src.size}"
    if src.sha256 and sha != src.sha256.lower():
        return f"sha256 {sha} != pinned {src.sha256}"
    if src.md5 and md5 != src.md5.lower():
        return f"md5 {md5} != pinned {src.md5}"
    return None


def fetch(
    name: str,
    dest_dir: str | Path | None = None,
    *,
    sources: Mapping[str, Source] = SOURCES,
    timeout_s: float = 60.0,
) -> Path:
    """Path to a verified local copy of ``name``, downloading it into ``dest_dir`` if needed.

    A cached file is reused only if it still matches its checksums; otherwise it is
    downloaded again. Downloads go to a ``.part`` file that is renamed into place only
    after verification, so an interrupted or tampered download never looks complete.
    """
    if name not in sources:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(sorted(sources))}")
    src = sources[name]
    if not (src.sha256 or src.md5):
        raise ValueError(f"{name}: no checksum pinned, refusing to fetch an unverifiable file")
    dest = Path(dest_dir) if dest_dir is not None else default_cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / src.filename
    if target.is_file() and _mismatch(target, src) is None:
        return target
    part = target.with_name(target.name + ".part")
    try:
        request = urllib.request.Request(src.url, headers={"User-Agent": "kappa-eval/0.1"})
        with (
            urllib.request.urlopen(request, timeout=timeout_s) as response,
            open(part, "wb") as out,
        ):
            shutil.copyfileobj(response, out, 1 << 20)
        problem = _mismatch(part, src)
        if problem is not None:
            raise DatasetIntegrityError(f"{name} from {src.url}: {problem}")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return target
