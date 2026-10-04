"""Parser and deterministic inspection output for CV22 model containers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .ranges import ByteRange, RangeBoundsError, RangeLedger
from .reader import BinaryReader, BoundsError
from .schema import ArtifactSpan, CV22Container, ContainerHeader, SplitRecord


CV22_MAGIC = 0x02010007
HEADER_SIZE = 12
DIRECTORY_RECORD_SIZE = 156
DIRECTORY_RAW_WORDS = 7
DIRECTORY_NAME_SIZE = 128


class ContainerFormatError(ValueError):
    """The source cannot be represented by the evidenced container schema."""


def _read_name(reader: BinaryReader, offset: int) -> str:
    raw_name = reader.read_bytes(offset, DIRECTORY_NAME_SIZE)
    if b"\x00" not in raw_name:
        raise ContainerFormatError(
            f"directory name at offset {offset} has no NUL terminator"
        )
    encoded_name = raw_name.partition(b"\x00")[0]
    if not encoded_name:
        raise ContainerFormatError(f"directory name at offset {offset} is empty")
    try:
        return encoded_name.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ContainerFormatError(
            f"directory name at offset {offset} is not ASCII"
        ) from exc


def parse_container(data: Any) -> CV22Container:
    """Parse the evidenced top-level layout without interpreting payloads."""

    reader = BinaryReader(data)
    if reader.size < HEADER_SIZE:
        raise ContainerFormatError(
            f"truncated header: need {HEADER_SIZE} bytes, have {reader.size}"
        )

    magic = reader.read_u32le(0)
    if magic != CV22_MAGIC:
        raise ContainerFormatError(
            f"unexpected container magic 0x{magic:08x}; expected 0x{CV22_MAGIC:08x}"
        )

    raw_word_04 = reader.read_u32le(4)
    record_count = reader.read_u32le(8)
    directory_size = record_count * DIRECTORY_RECORD_SIZE
    directory_end = HEADER_SIZE + directory_size
    if directory_end > reader.size:
        raise ContainerFormatError(
            f"truncated directory: {record_count} records require "
            f"{directory_end} bytes, file has {reader.size}"
        )

    pending: list[tuple[int, int, tuple[int, ...], str]] = []
    try:
        for record_index in range(record_count):
            directory_offset = HEADER_SIZE + (record_index * DIRECTORY_RECORD_SIZE)
            raw_words = tuple(
                reader.read_u32le(directory_offset + (word_index * 4))
                for word_index in range(DIRECTORY_RAW_WORDS)
            )
            name = _read_name(reader, directory_offset + 28)
            pending.append((record_index, directory_offset, raw_words, name))
    except BoundsError as exc:
        raise ContainerFormatError(f"truncated directory: {exc}") from exc

    payload_cursor = directory_end
    records: list[SplitRecord] = []
    ledger = RangeLedger(reader.size).add(ByteRange(0, HEADER_SIZE, "header"))
    for record_index in range(record_count):
        directory_offset = HEADER_SIZE + (record_index * DIRECTORY_RECORD_SIZE)
        ledger = ledger.add(
            ByteRange(
                directory_offset,
                directory_offset + DIRECTORY_RECORD_SIZE,
                f"directory[{record_index}]",
            )
        )

    for record_index, directory_offset, raw_values, name in pending:
        raw_words = (
            raw_values[0],
            raw_values[1],
            raw_values[2],
            raw_values[3],
            raw_values[4],
            raw_values[5],
            raw_values[6],
        )
        payload_size = raw_words[6]
        if payload_size <= 0:
            raise ContainerFormatError(
                f"payload for record {record_index} has invalid size {payload_size}"
            )
        payload_end = payload_cursor + payload_size
        if payload_end > reader.size:
            raise ContainerFormatError(
                f"payload for record {record_index} [{payload_cursor}, "
                f"{payload_end}) exceeds file size {reader.size}"
            )
        payload = ArtifactSpan(payload_cursor, payload_size)
        records.append(
            SplitRecord(
                index=record_index,
                directory_offset=directory_offset,
                raw_words=raw_words,
                name=name,
                payload=payload,
            )
        )
        try:
            ledger = ledger.add(
                ByteRange(
                    payload.offset,
                    payload.end,
                    f"payload[{record_index}]",
                )
            )
        except RangeBoundsError as exc:
            raise ContainerFormatError(f"invalid payload range: {exc}") from exc
        payload_cursor = payload_end

    return CV22Container(
        file_size=reader.size,
        header=ContainerHeader(magic, raw_word_04, record_count),
        records=tuple(records),
        structural_report=ledger.report(),
    )


def _interval_dict(interval: Any) -> dict[str, int]:
    return {
        "offset": interval.start,
        "size": interval.length,
        "end": interval.end,
    }


def inspect_path(path: str | Path) -> dict[str, Any]:
    """Return deterministic JSON-ready evidence for one model artifact."""

    source = Path(path)
    data = source.read_bytes()
    container = parse_container(data)
    records: list[dict[str, Any]] = []
    for record in container.records:
        payload_bytes = memoryview(data)[record.payload.offset : record.payload.end]
        records.append(
            {
                "index": record.index,
                "directory_offset": record.directory_offset,
                "name_offset": record.name_offset,
                "name_size": DIRECTORY_NAME_SIZE,
                "name": record.name,
                "raw_word_offsets": list(record.raw_word_offsets),
                "raw_words": list(record.raw_words),
                "raw_words_hex": [f"0x{value:08x}" for value in record.raw_words],
                "payload": {
                    "offset": record.payload.offset,
                    "size": record.payload.size,
                    "end": record.payload.end,
                },
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            }
        )

    return {
        "schema": "verkeye.cv22.container.v1",
        "artifact": {
            "path": str(source.resolve()),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "container": {
            "header": {
                "offset": 0,
                "size": HEADER_SIZE,
                "magic": container.header.magic,
                "magic_hex": f"0x{container.header.magic:08x}",
                "raw_word_04": container.header.raw_word_04,
                "raw_word_04_hex": f"0x{container.header.raw_word_04:08x}",
                "record_count": container.header.record_count,
            },
            "directory": {
                "offset": container.directory_offset,
                "record_size": container.directory_record_size,
                "size": container.directory_size,
                "end": container.payload_offset,
            },
            "payload_size_word_index": 6,
            "structural_coverage_complete": container.structural_report.complete,
            "top_level_unknown_bytes": container.top_level_unknown_bytes,
            "top_level_unknown_ranges": [
                _interval_dict(interval)
                for interval in container.structural_report.unknown
            ],
            "opaque_payload_bytes": container.opaque_payload_bytes,
            "interpretation_boundary": (
                "payload spans are structurally bounded but remain opaque; "
                "raw directory words 0 through 5 are intentionally unnamed"
            ),
        },
        "records": records,
    }

