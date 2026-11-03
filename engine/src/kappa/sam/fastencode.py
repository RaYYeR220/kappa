"""A compiled encoder for SAM 3.1's ``one_bit`` mask payloads.

The reference encoder (``meta_sam_parser._mask_codec._encode_raster``) is pure Python,
about 6 ms per mouse-sized mask. The self-hosted tracker encodes every mask of a
15,000-frame video, so this is the same adaptive binary range coder and 12-pixel
context model, compiled with numba, plus the canonical base85 packing. It is checked
byte for byte against the reference encoder in the tests and round-trips through
:mod:`kappa.sam.fastmask`.

The module depends on numpy and numba only, so the GPU service can ship it as a
single file without the rest of the engine.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_TOP = 1 << 24
_BOTTOM = 1 << 16
_MASK_32 = 0xFFFFFFFF
_INCREMENT = 14
_COUNT_LIMIT = 4096
_CONTEXT_COUNT = 1 << 12
_PADDING = 2
_RADIX = 85
_PREFIX_LENGTH = 5

_EXCLUDED = frozenset({'"', "\\", ",", ";", "<", ">", "|"})
_ALPHABET = "".join(c for code in range(0x21, 0x7F) if (c := chr(code)) not in _EXCLUDED)
_ALPHABET_BYTES = np.frombuffer(_ALPHABET[:_RADIX].encode("ascii"), dtype=np.uint8).copy()


@njit(cache=True)
def _encode_bits(raster: np.ndarray, width: int, height: int) -> np.ndarray:
    """Range-code a flat 0/1 raster (row-major, ``height * width``) into bytes."""
    counts = np.ones(_CONTEXT_COUNT * 2, dtype=np.int64)
    pw = width + 2 * _PADDING
    padded = np.zeros(pw * (height + _PADDING), dtype=np.int64)
    out = np.empty(max(64, (width * height) // 4 + 16), dtype=np.uint8)
    n = 0
    low = 0
    rng = _MASK_32
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
            bit = np.int64(raster[y * width + x])
            if bit != 0 and bit != 1:
                raise ValueError("mask raster is not binary")
            scaled = rng // (zero + one)
            if bit == 0:
                rng = scaled * zero
            else:
                low = (low + scaled * zero) & _MASK_32
                rng = scaled * one
            while True:
                if (low ^ ((low + rng) & _MASK_32)) < _TOP:
                    pass
                elif rng < _BOTTOM:
                    rng = ((-low) & _MASK_32) & (_BOTTOM - 1)
                else:
                    break
                if n >= out.shape[0]:
                    grown = np.empty(out.shape[0] * 2, dtype=np.uint8)
                    grown[:n] = out[:n]
                    out = grown
                out[n] = (low >> 24) & 0xFF
                n += 1
                low = (low << 8) & _MASK_32
                rng = (rng << 8) & _MASK_32
            padded[p] = bit
            counts[off + bit] = (counts[off + bit] + _INCREMENT) & 0xFFFF
            if counts[off] + counts[off + 1] >= _COUNT_LIMIT:
                c0 = counts[off] >> 1
                c1 = counts[off + 1] >> 1
                counts[off] = c0 if c0 else 1
                counts[off + 1] = c1 if c1 else 1
    if n + 4 > out.shape[0]:
        grown = np.empty(n + 4, dtype=np.uint8)
        grown[:n] = out[:n]
        out = grown
    for _ in range(4):
        out[n] = (low >> 24) & 0xFF
        n += 1
        low = (low << 8) & _MASK_32
    return out[:n].copy()


@njit(cache=True)
def _pack_bytes(data: np.ndarray, alphabet: np.ndarray) -> np.ndarray:
    """Canonical base85: a 5-digit length prefix, then 5 digits per 4 bytes, tail trimmed."""
    length = data.shape[0]
    remainder = length % 4
    full = length - remainder
    size = _PREFIX_LENGTH + (full // 4) * 5 + (remainder + 1 if remainder else 0)
    out = np.empty(size, dtype=np.uint8)
    rest = length
    for i in range(_PREFIX_LENGTH - 1, -1, -1):
        out[i] = alphabet[rest % _RADIX]
        rest //= _RADIX
    if rest != 0:
        raise ValueError("mask payload is too large")
    pos = _PREFIX_LENGTH
    digits = np.empty(5, dtype=np.int64)
    offset = 0
    while offset < full:
        value = (
            (np.int64(data[offset]) << 24)
            | (np.int64(data[offset + 1]) << 16)
            | (np.int64(data[offset + 2]) << 8)
            | np.int64(data[offset + 3])
        )
        for i in range(4, -1, -1):
            digits[i] = value % _RADIX
            value //= _RADIX
        for i in range(5):
            out[pos + i] = alphabet[digits[i]]
        pos += 5
        offset += 4
    if remainder:
        value = np.int64(0)
        for i in range(4):
            value = (value << 8) | (np.int64(data[offset + i]) if i < remainder else 0)
        for i in range(4, -1, -1):
            digits[i] = value % _RADIX
            value //= _RADIX
        for i in range(remainder + 1):
            out[pos + i] = alphabet[digits[i]]
    return out


def encode_mask(raster: np.ndarray) -> str:
    """The ``one_bit`` payload (``"!..."``) of a ``(height, width)`` 0/1 raster.

    Byte for byte what ``meta_sam_parser._mask_codec._encode_raster`` returns for the
    same raster, so the reference parser accepts it as canonical.
    """
    raster = np.asarray(raster)
    if raster.ndim != 2 or raster.shape[0] == 0 or raster.shape[1] == 0:
        raise ValueError(f"expected a non-empty 2-D raster, got shape {raster.shape}")
    height, width = raster.shape
    flat = np.ascontiguousarray(raster, dtype=np.uint8).reshape(-1)
    packed = _pack_bytes(_encode_bits(flat, width, height), _ALPHABET_BYTES)
    return "!" + packed.tobytes().decode("ascii")
