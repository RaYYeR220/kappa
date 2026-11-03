import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from meta_sam_parser._mask_codec import _encode_raster, _pack

from kappa.sam.fastencode import _ALPHABET_BYTES, _pack_bytes, encode_mask
from kappa.sam.fastmask import decode_mask


def _reference(mask: np.ndarray) -> str:
    h, w = mask.shape
    return _encode_raster(mask.reshape(-1).tolist(), w, h)


@settings(max_examples=150, deadline=None)
@given(
    st.integers(1, 48),
    st.integers(1, 48),
    st.floats(0.0, 1.0),
    st.integers(0, 2**31 - 1),
)
def test_compiled_encoder_matches_reference_byte_for_byte(h, w, density, seed):
    rng = np.random.default_rng(seed)
    mask = (rng.random((h, w)) < density).astype(np.uint8)
    payload = encode_mask(mask)
    assert payload == _reference(mask)
    assert np.array_equal(decode_mask("one_bit", payload, w, h), mask)


@settings(max_examples=40, deadline=None)
@given(st.integers(8, 160), st.integers(8, 160), st.integers(0, 2**31 - 1))
def test_blob_masks_match_reference_and_round_trip(h, w, seed):
    rng = np.random.default_rng(seed)
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    ry, rx = rng.uniform(2, h), rng.uniform(2, w)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1).astype(np.uint8)
    payload = encode_mask(mask)
    assert payload == _reference(mask)
    raw = decode_mask_to_raster(SegmentationMask(encoding="one_bit", payload=payload, width=w, height=h))
    assert np.array_equal(np.frombuffer(raw, dtype=np.uint8).reshape(h, w), mask)


@settings(max_examples=100, deadline=None)
@given(st.binary(min_size=0, max_size=64))
def test_base85_packing_matches_reference(data):
    packed = _pack_bytes(np.frombuffer(data, dtype=np.uint8), _ALPHABET_BYTES).tobytes().decode("ascii")
    assert packed == _pack(data)


def test_bool_rasters_and_large_masks():
    yy, xx = np.mgrid[0:600, 400:0:-1]
    mask = ((yy * 3 + xx * 7) % 11 < 5) | ((yy - 300) ** 2 + (xx - 200) ** 2 < 150**2)
    assert encode_mask(mask) == _reference(mask.astype(np.uint8))


def test_rejects_non_binary_and_empty():
    with pytest.raises(ValueError):
        encode_mask(np.full((3, 3), 2, dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_mask(np.zeros((0, 4), dtype=np.uint8))
