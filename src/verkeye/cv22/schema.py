"""Immutable top-level schema for an Ambarella CV22 model container.

Only fields whose placement is directly supported by the recovered artifact
are named.  Unresolved directory values remain an ordered tuple of raw words.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ranges import RangeReport


@dataclass(frozen=True, slots=True)
class ArtifactSpan:
    """A half-open span in the source artifact."""

    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True, slots=True)
class ContainerHeader:
    """The three evidenced 32-bit words at the start of the container."""

    magic: int
    raw_word_04: int
    record_count: int


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """One fixed-width directory record and its serialized payload span."""

    index: int
    directory_offset: int
    raw_words: tuple[int, int, int, int, int, int, int]
    name: str
    payload: ArtifactSpan

    @property
    def name_offset(self) -> int:
        return self.directory_offset + 28

    @property
    def raw_word_offsets(self) -> tuple[int, ...]:
        return tuple(self.directory_offset + (index * 4) for index in range(7))


@dataclass(frozen=True, slots=True)
class CV22Container:
    """Lossless top-level model-container map."""

    file_size: int
    header: ContainerHeader
    records: tuple[SplitRecord, ...]
    structural_report: RangeReport

    directory_offset: int = 12
    directory_record_size: int = 156

    @property
    def directory_size(self) -> int:
        return self.header.record_count * self.directory_record_size

    @property
    def payload_offset(self) -> int:
        return self.directory_offset + self.directory_size

    @property
    def opaque_payload_bytes(self) -> int:
        return sum(record.payload.size for record in self.records)

    @property
    def top_level_unknown_bytes(self) -> int:
        return sum(interval.length for interval in self.structural_report.unknown)

