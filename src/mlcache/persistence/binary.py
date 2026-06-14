"""Binary encoders for durable MLCache storage."""

from __future__ import annotations

import struct
from collections.abc import Sequence


def encode_float64_vector(values: Sequence[float]) -> bytes:
    """Encode floats as big-endian float64 bytes for portable BYTEA storage."""

    floats = tuple(float(value) for value in values)
    if not floats:
        return b""
    return struct.pack(f"!{len(floats)}d", *floats)


def decode_float64_vector(blob: bytes | memoryview | None) -> tuple[float, ...]:
    """Decode bytes produced by `encode_float64_vector`."""

    if blob is None:
        return ()
    data = bytes(blob)
    if not data:
        return ()
    if len(data) % 8 != 0:
        raise ValueError("float64 vector blob length must be divisible by 8")
    count = len(data) // 8
    return tuple(float(value) for value in struct.unpack(f"!{count}d", data))


__all__ = ["decode_float64_vector", "encode_float64_vector"]
