"""Bounds-checked primitives for recovered Ambarella CV22 artifacts."""

from .container import CV22_MAGIC, ContainerFormatError, inspect_path, parse_container
from .reader import BinaryReader, BoundsError
from .ranges import (
    ByteRange,
    Interval,
    RangeBoundsError,
    RangeCollision,
    RangeCoverageError,
    RangeLedger,
    RangeOverlapError,
    RangeReport,
)

__all__ = [
    "BinaryReader",
    "BoundsError",
    "CV22_MAGIC",
    "ByteRange",
    "Interval",
    "RangeBoundsError",
    "RangeCollision",
    "RangeCoverageError",
    "RangeLedger",
    "RangeOverlapError",
    "RangeReport",
    "ContainerFormatError",
    "inspect_path",
    "parse_container",
]
