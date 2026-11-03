"""A compiled decoder for SAM's one_bit masks.

The official parser decodes pixel by pixel in Python, then re-encodes to check itself, which takes about a second
for a large mask. That is fine once and fatal in a live loop or across a whole video. This is the same range decoder
(see meta_sam_parser._mask_codec) compiled with numba; tests/test_core.py checks it against the official parser.
"""
from __future__ import annotations

import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from meta_sam_parser._mask_codec import _unpack

try:
    from numba import njit
except ImportError:  # still correct without numba, only slower
    njit = None


def _decode(data, width, height):
    mask32 = 0xFFFFFFFF
    top, bottom = 1 << 24, 1 << 16
    padding = 2
    counts = np.ones(1 << 13, dtype=np.int64)
    padded_width = width + 2 * padding
    padded = np.zeros(padded_width * (height + padding), dtype=np.int64)
    output = np.zeros((height, width), dtype=np.uint8)
    low, rng, code, position = 0, mask32, 0, 0
    n = data.shape[0]
    if n < 4:
        raise ValueError("mask payload too short")
    for _ in range(4):
        code = ((code << 8) | int(data[position])) & mask32
        position += 1
    for y in range(height):
        row = (y + padding) * padded_width + padding
        for x in range(width):
            p = row + x
            prior, earlier = p - padded_width, p - 2 * padded_width
            context = (padded[p - 1] | (padded[p - 2] << 1) | (padded[prior - 2] << 2) | (padded[prior - 1] << 3)
                       | (padded[prior] << 4) | (padded[prior + 1] << 5) | (padded[prior + 2] << 6)
                       | (padded[earlier - 2] << 7) | (padded[earlier - 1] << 8) | (padded[earlier] << 9)
                       | (padded[earlier + 1] << 10) | (padded[earlier + 2] << 11))
            offset = int(context) << 1
            zero, one = int(counts[offset]), int(counts[offset + 1])
            total = zero + one
            scaled = rng // total
            if scaled == 0:
                raise ValueError("invalid mask coding state")
            value = ((code - low) & mask32) // scaled
            if value >= total:
                value = total - 1
            if value < zero:
                bit, cumulative, frequency = 0, 0, zero
            else:
                bit, cumulative, frequency = 1, zero, one
            low = (low + scaled * cumulative) & mask32
            rng = scaled * frequency
            while True:
                if (low ^ ((low + rng) & mask32)) < top:
                    pass
                elif rng < bottom:
                    rng = ((-low) & mask32) & (bottom - 1)
                else:
                    break
                if position >= n:
                    raise ValueError("mask payload ended early")
                code = ((code << 8) | int(data[position])) & mask32
                position += 1
                low = (low << 8) & mask32
                rng = (rng << 8) & mask32
            padded[p] = bit
            output[y, x] = bit
            counts[offset + bit] = (counts[offset + bit] + 14) & 0xFFFF
            if counts[offset] + counts[offset + 1] >= 4096:
                counts[offset] = max(counts[offset] >> 1, 1)
                counts[offset + 1] = max(counts[offset + 1] >> 1, 1)
    return output


if njit is not None:
    _decode = njit(cache=True)(_decode)


def decode(payload: str, height: int, width: int) -> np.ndarray:
    """A SAM mask payload as a boolean array of shape (height, width)."""
    if njit is not None and payload.startswith("!"):
        data = np.frombuffer(_unpack(payload[1:]), dtype=np.uint8)
        return _decode(data, width, height).astype(bool)
    encoding = "one_bit" if payload.startswith("!") else "lossless"
    raster = decode_mask_to_raster(SegmentationMask(encoding, payload, width, height))
    return np.frombuffer(raster, dtype=np.uint8).reshape(height, width).astype(bool)
