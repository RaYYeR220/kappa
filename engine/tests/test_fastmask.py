import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from meta_sam_parser._mask_codec import _encode_raster

from kappa.sam.fastmask import decode_mask


def _reference(payload, w, h):
    raw = decode_mask_to_raster(SegmentationMask(encoding="one_bit", payload=payload, width=w, height=h))
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w)


@settings(max_examples=60, deadline=None)
@given(
    st.integers(1, 40),
    st.integers(1, 40),
    st.floats(0.0, 1.0),
    st.integers(0, 2**31 - 1),
)
def test_compiled_decoder_matches_reference_bit_for_bit(h, w, density, seed):
    rng = np.random.default_rng(seed)
    mask = (rng.random((h, w)) < density).astype(np.uint8)
    payload = _encode_raster(mask.reshape(-1).tolist(), w, h)
    assert np.array_equal(decode_mask("one_bit", payload, w, h), _reference(payload, w, h))
    assert np.array_equal(decode_mask("one_bit", payload, w, h), mask)


def test_blob_shaped_masks_round_trip():
    yy, xx = np.mgrid[0:120, 0:90]
    mask = ((((yy - 60) / 55) ** 2 + ((xx - 45) / 30) ** 2) <= 1).astype(np.uint8)
    payload = _encode_raster(mask.reshape(-1).tolist(), 90, 120)
    assert np.array_equal(decode_mask("one_bit", payload, 90, 120), mask)
