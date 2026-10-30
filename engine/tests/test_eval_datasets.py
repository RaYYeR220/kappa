import hashlib

import pytest

from kappa.eval.datasets import (
    SOURCES,
    DatasetIntegrityError,
    Source,
    default_cache_dir,
    fetch,
    sha256_file,
)


@pytest.fixture
def remote(tmp_path):
    """A 'remote' file served over file:// so fetch runs its real download path offline."""
    src = tmp_path / "remote" / "labels.csv"
    src.parent.mkdir()
    src.write_bytes(b"a;b\n1;2\n")
    return src


def _source(path, **overrides):
    data = path.read_bytes()
    fields = {
        "url": path.as_uri(),
        "filename": "labels.csv",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }
    fields.update(overrides)
    return Source(**fields)


def test_fetch_downloads_and_verifies(tmp_path, remote):
    sources = {"demo": _source(remote)}
    got = fetch("demo", tmp_path / "cache", sources=sources)
    assert got == tmp_path / "cache" / "labels.csv"
    assert got.read_bytes() == remote.read_bytes()
    assert sha256_file(got) == sources["demo"].sha256


def test_fetch_reuses_a_verified_cache(tmp_path, remote):
    sources = {"demo": _source(remote)}
    fetch("demo", tmp_path / "cache", sources=sources)
    remote.unlink()  # a second download would now fail
    assert fetch("demo", tmp_path / "cache", sources=sources).is_file()


def test_fetch_replaces_a_corrupted_cache(tmp_path, remote):
    sources = {"demo": _source(remote)}
    cached = fetch("demo", tmp_path / "cache", sources=sources)
    cached.write_bytes(b"tampered")
    assert fetch("demo", tmp_path / "cache", sources=sources).read_bytes() == remote.read_bytes()


def test_fetch_rejects_a_checksum_mismatch_and_leaves_nothing(tmp_path, remote):
    sources = {"demo": _source(remote, sha256="0" * 64)}
    with pytest.raises(DatasetIntegrityError):
        fetch("demo", tmp_path / "cache", sources=sources)
    assert list((tmp_path / "cache").iterdir()) == []


def test_fetch_accepts_a_published_md5(tmp_path, remote):
    md5 = hashlib.md5(remote.read_bytes()).hexdigest()
    sources = {"demo": _source(remote, sha256=None, md5=md5)}
    assert fetch("demo", tmp_path / "cache", sources=sources).is_file()


def test_fetch_refuses_unpinned_and_unknown_sources(tmp_path, remote):
    with pytest.raises(ValueError):
        fetch("demo", tmp_path, sources={"demo": _source(remote, sha256=None)})
    with pytest.raises(KeyError):
        fetch("nope", tmp_path, sources={})


def test_default_cache_dir_honours_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("KAPPA_DATA_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path


def test_manifest_pins_every_source():
    assert SOURCES["sturman_oft_labels"].url == (
        "https://raw.githubusercontent.com/ETHZ-INS/DLCAnalyzer/master/data/OFT/Labels/AllLabDataOFT_final.csv"
    )
    assert (
        SOURCES["sturman_oft_labels"].sha256
        == "2a690c586f00af4574be5555e63ff66698a3149ef8c70e050876f138e7fd5fd3"
    )
    videos = [n for n in SOURCES if n.startswith("sturman_video_OFT_")]
    assert len(videos) == 20
    assert SOURCES["sturman_video_OFT_23"].sha256 is not None
    for name, src in SOURCES.items():
        assert src.url.startswith("https://"), name
        assert src.sha256 or src.md5, name
        assert src.licence, name
