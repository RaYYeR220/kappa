"""A compiled decoder for SAM 3.1's ``one_bit`` mask payloads.

The reference parser decodes masks in pure Python, about 12 ms for a mouse-sized
mask; a ten-minute clip has 15,000 of them. This is the same adaptive range decoder
and context model, compiled with numba, and it is checked bit for bit against the
reference decoder in the tests. Payloads are still unpacked (base85, with the
reference's canonical-form check) by the reference code; only the per-pixel loop
is compiled.
"""

from __future__ import annotations

import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from meta_sam_parser._mask_codec import _unpack
from numba import njit

_TOP = 1 << 24
_BOTTOM = 1 << 16
_MASK_32 = 0xFFFFFFFF
_INCREMENT = 14
_COUNT_LIMIT = 4096
_CONTEXT_COUNT = 1 << 12
_PADDING = 2


@njit(cache=True)
def _decode_bits(data: np.ndarray, width: int, height: int) -> np.ndarray:
    n = data.shape[0]
    if n < 4:
        raise ValueError("mask payload is missing its finalization")
    pos = 0
    low = 0
    rng = _MASK_32
    code = 0
    for _ in range(4):
        code = ((code << 8) | data[pos]) & _MASK_32
        pos += 1
    counts = np.ones(_CONTEXT_COUNT * 2, dtype=np.int64)
    pw = width + 2 * _PADDING
    padded = np.zeros(pw * (height + _PADDING), dtype=np.int64)
    out = np.zeros(width * height, dtype=np.uint8)
    for y in range(height):
        row = (y + _PADDING) * pw + _PADDING
        for x in range(width):
            p = row + x
            prior = p - pw
            earlier = p - 2 * pw
            ctx = (
                padded[p - 1]
                | (padded[p - 2] << 1)
                | (padded[prior - 2] << 2)
                | (padded[prior - 1] << 3)
                | (padded[prior] << 4)
                | (padded[prior + 1] << 5)
                | (padded[prior + 2] << 6)
                | (padded[earlier - 2] << 7)
                | (padded[earlier - 1] << 8)
                | (padded[earlier] << 9)
                | (padded[earlier + 1] << 10)
                | (padded[earlier + 2] << 11)
            )
            off = ctx << 1
            zero = counts[off]
            one = counts[off + 1]
            total = zero + one
            scaled = rng // total
            if scaled == 0:
                raise ValueError("mask payload has invalid coding state")
            value = ((code - low) & _MASK_32) // scaled
            if value >= total:
                value = total - 1
            if value < zero:
                bit = 0
                low = (low + scaled * 0) & _MASK_32
                rng = scaled * zero
            else:
                bit = 1
                low = (low + scaled * zero) & _MASK_32
                rng = scaled * one
            while True:
                if (low ^ ((low + rng) & _MASK_32)) < _TOP:
                    pass
                elif rng < _BOTTOM:
                    rng = ((-low) & _MASK_32) & (_BOTTOM - 1)
                else:
                    break
                if pos >= n:
                    raise ValueError("mask payload ended before decoding completed")
                code = ((code << 8) | data[pos]) & _MASK_32
                pos += 1
                low = (low << 8) & _MASK_32
                rng = (rng << 8) & _MASK_32
            padded[p] = bit
            out[y * width + x] = bit
            counts[off + bit] = (counts[off + bit] + _INCREMENT) & 0xFFFF
            if counts[off] + counts[off + 1] >= _COUNT_LIMIT:
                c0 = counts[off] >> 1
                c1 = counts[off + 1] >> 1
                counts[off] = c0 if c0 else 1
                counts[off + 1] = c1 if c1 else 1
    return out


def decode_mask(encoding: str, payload: str, width: int, height: int) -> np.ndarray:
    """A mask as a ``(height, width)`` uint8 array of 0/1."""
    if encoding == "one_bit" and payload.startswith("!"):
        data = np.frombuffer(_unpack(payload[1:]), dtype=np.uint8)
        return _decode_bits(data, width, height).reshape(height, width)
    raster = decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=width, height=height))
    return np.frombuffer(raster, dtype=np.uint8).reshape(height, width)
