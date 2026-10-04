"""Small, fail-closed binary-reading primitives.

The reader deliberately exposes a read-only ``memoryview`` for zero-copy
parsing while making every span validation explicit.  Parsers built on this
module cannot receive a silently truncated slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index
from typing import Any


class BoundsError(ValueError):
    """A requested byte span is outside the artifact."""


@dataclass(frozen=True, slots=True, init=False)
class BinaryReader:
    """Read primitive values from an immutable byte-oriented view."""

    _view: memoryview = field(repr=False)

    def __init__(self, data: Any) -> None:
        try:
            view = memoryview(data)
            if view.ndim != 1 or view.format != "B":
                view = view.cast("B")
            view = view.toreadonly()
        except (TypeError, ValueError) as exc:
            raise TypeError("data must expose a contiguous byte buffer") from exc
        object.__setattr__(self, "_view", view)

    @property
    def size(self) -> int:
        return len(self._view)

    def _checked_span(self, offset: int, size: int) -> tuple[int, int]:
        try:
            start = index(offset)
            length = index(size)
        except TypeError as exc:
            raise BoundsError("offset and size must be integers") from exc

        if start < 0 or length < 0:
            raise BoundsError(
                f"negative byte span: offset={start}, size={length}, file_size={self.size}"
            )
        end = start + length
        if start > self.size or end > self.size:
            raise BoundsError(
                f"byte span [{start}, {end}) exceeds file size {self.size}"
            )
        return start, end

    def read_slice(self, offset: int, size: int) -> memoryview:
        """Return an exact, read-only view of ``[offset, offset + size)``."""

        start, end = self._checked_span(offset, size)
        return self._view[start:end]

    def read_bytes(self, offset: int, size: int) -> bytes:
        """Return an exact byte copy of a checked span."""

        return self.read_slice(offset, size).tobytes()

    def read_u8(self, offset: int) -> int:
        return self._read_uint(offset, 1)

    def read_u16le(self, offset: int) -> int:
        return self._read_uint(offset, 2)

    def read_u32le(self, offset: int) -> int:
        return self._read_uint(offset, 4)

    def read_u64le(self, offset: int) -> int:
        return self._read_uint(offset, 8)

    def _read_uint(self, offset: int, size: int) -> int:
        return int.from_bytes(self.read_slice(offset, size), byteorder="little")

    def read_fixed_string(
        self,
        offset: int,
        size: int,
        *,
        encoding: str = "ascii",
    ) -> str:
        """Decode a fixed-width, NUL-terminated string field strictly."""

        field = self.read_bytes(offset, size)
        return field.partition(b"\x00")[0].decode(encoding, errors="strict")

    def align(self, offset: int, alignment: int) -> int:
        """Round an in-file offset upward without crossing EOF."""

        try:
            boundary = index(alignment)
        except TypeError as exc:
            raise ValueError("alignment must be a positive integer") from exc
        if boundary <= 0:
            raise ValueError("alignment must be a positive integer")

        start, _ = self._checked_span(offset, 0)
        aligned = ((start + boundary - 1) // boundary) * boundary
        if aligned > self.size:
            raise BoundsError(
                f"aligned offset {aligned} exceeds file size {self.size}"
            )
        return aligned

