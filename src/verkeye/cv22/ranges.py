"""Lossless byte-range accounting for partially understood binary formats."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from operator import index


class RangeBoundsError(ValueError):
    """A declared range is malformed or outside the artifact."""


class RangeOverlapError(ValueError):
    """Two interpreted ranges overlap without an exact alias declaration."""


class RangeCoverageError(ValueError):
    """One or more artifact bytes remain unaccounted for."""


def _integer(value: int, name: str) -> int:
    try:
        return index(value)
    except TypeError as exc:
        raise RangeBoundsError(f"{name} must be an integer") from exc


@dataclass(frozen=True, slots=True, order=True)
class Interval:
    """A half-open byte interval."""

    start: int
    end: int

    def __post_init__(self) -> None:
        start = _integer(self.start, "start")
        end = _integer(self.end, "end")
        if start < 0 or end < start:
            raise RangeBoundsError(f"invalid interval [{start}, {end})")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class ByteRange:
    """An interpretation assigned to a non-empty artifact byte span."""

    start: int
    end: int
    label: str
    alias_group: str | None = None

    def __post_init__(self) -> None:
        start = _integer(self.start, "start")
        end = _integer(self.end, "end")
        if start < 0 or end <= start:
            raise RangeBoundsError(f"invalid byte range [{start}, {end})")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("range label must be non-empty")
        if self.alias_group is not None and (
            not isinstance(self.alias_group, str) or not self.alias_group.strip()
        ):
            raise ValueError("alias_group must be None or a non-empty string")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def interval(self) -> Interval:
        return Interval(self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class RangeCollision:
    """The exact intersection of two declared ranges."""

    interval: Interval
    left: ByteRange
    right: ByteRange


@dataclass(frozen=True, slots=True)
class RangeReport:
    """Complete accounting result for a range ledger."""

    file_size: int
    covered: tuple[Interval, ...]
    aliased: tuple[RangeCollision, ...]
    overlapping: tuple[RangeCollision, ...]
    unknown: tuple[Interval, ...]

    @property
    def valid(self) -> bool:
        return not self.overlapping

    @property
    def complete(self) -> bool:
        return not self.unknown

    def require_valid(self) -> RangeReport:
        if self.overlapping:
            collision = self.overlapping[0]
            raise RangeOverlapError(
                "illegal overlap "
                f"[{collision.interval.start}, {collision.interval.end}) between "
                f"{collision.left.label!r} and {collision.right.label!r}"
            )
        return self

    def require_complete(self) -> RangeReport:
        self.require_valid()
        if self.unknown:
            spans = ", ".join(
                f"[{item.start}, {item.end})" for item in self.unknown
            )
            raise RangeCoverageError(f"unknown byte interval(s): {spans}")
        return self


@dataclass(frozen=True, slots=True)
class RangeLedger:
    """An immutable set of labeled byte ranges for one artifact."""

    file_size: int
    ranges: tuple[ByteRange, ...] = ()

    def __post_init__(self) -> None:
        file_size = _integer(self.file_size, "file_size")
        if file_size < 0:
            raise RangeBoundsError("file_size cannot be negative")
        ranges = tuple(self.ranges)
        for byte_range in ranges:
            self._check_range(byte_range, file_size)
        object.__setattr__(self, "file_size", file_size)
        object.__setattr__(self, "ranges", ranges)

    @staticmethod
    def _check_range(byte_range: ByteRange, file_size: int) -> None:
        if not isinstance(byte_range, ByteRange):
            raise TypeError("ledger entries must be ByteRange instances")
        if byte_range.end > file_size:
            raise RangeBoundsError(
                f"range {byte_range.label!r} [{byte_range.start}, "
                f"{byte_range.end}) exceeds file size {file_size}"
            )

    def add(self, byte_range: ByteRange) -> RangeLedger:
        self._check_range(byte_range, self.file_size)
        return RangeLedger(self.file_size, self.ranges + (byte_range,))

    def report(self) -> RangeReport:
        ordered = tuple(sorted(self.ranges, key=lambda item: (item.start, item.end, item.label)))
        covered = self._covered_intervals(ordered)
        aliases: list[RangeCollision] = []
        overlaps: list[RangeCollision] = []

        for left, right in combinations(ordered, 2):
            if right.start >= left.end:
                continue
            intersection = Interval(right.start, min(left.end, right.end))
            exact_alias = (
                left.start == right.start
                and left.end == right.end
                and left.alias_group is not None
                and left.alias_group == right.alias_group
            )
            collision = RangeCollision(intersection, left, right)
            if exact_alias:
                aliases.append(collision)
            else:
                overlaps.append(collision)

        return RangeReport(
            file_size=self.file_size,
            covered=covered,
            aliased=tuple(aliases),
            overlapping=tuple(overlaps),
            unknown=self._unknown_intervals(covered),
        )

    @staticmethod
    def _covered_intervals(ranges: tuple[ByteRange, ...]) -> tuple[Interval, ...]:
        merged: list[Interval] = []
        for byte_range in ranges:
            if not merged or byte_range.start > merged[-1].end:
                merged.append(byte_range.interval)
                continue
            previous = merged[-1]
            merged[-1] = Interval(previous.start, max(previous.end, byte_range.end))
        return tuple(merged)

    def _unknown_intervals(
        self, covered: tuple[Interval, ...]
    ) -> tuple[Interval, ...]:
        unknown: list[Interval] = []
        cursor = 0
        for interval in covered:
            if cursor < interval.start:
                unknown.append(Interval(cursor, interval.start))
            cursor = interval.end
        if cursor < self.file_size:
            unknown.append(Interval(cursor, self.file_size))
        return tuple(unknown)

