"""Cross-correlate unresolved CV22 scheduler words with recovered ORC images.

The ORC images contain mixed code and data and are not assumed to use the
scheduler's instruction encoding.  This module therefore reports exact byte
matches and alignment only.  A match is never promoted to instruction
semantics.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class OpenRiscCorpusError(ValueError):
    """The constraint ledger or recovered corpus is malformed."""


_TRACKED_OPCODES = (0x07, 0x10, 0x1C, 0x1F)


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _find_all(data: bytes, needle: bytes) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        offset = data.find(needle, start)
        if offset < 0:
            return offsets
        offsets.append(offset)
        start = offset + 1


def _unresolved_words(ledger: Mapping[str, Any]) -> tuple[list[int], Counter[int]]:
    if ledger.get("schema") != "verkeye.compat.openrisc-constraint-ledger.v1":
        raise OpenRiscCorpusError("constraint ledger schema is not supported")
    occurrences = ledger.get("occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        raise OpenRiscCorpusError("constraint ledger has no occurrences")

    counts: Counter[int] = Counter()
    for index, occurrence in enumerate(occurrences):
        if not isinstance(occurrence, Mapping):
            raise OpenRiscCorpusError(f"occurrence {index} is not an object")
        coverage = occurrence.get("semantic_coverage")
        if not isinstance(coverage, Mapping) or coverage.get("status") != "unresolved":
            raise OpenRiscCorpusError(f"occurrence {index} is not unresolved")
        raw_word = occurrence.get("word")
        if not isinstance(raw_word, str):
            raise OpenRiscCorpusError(f"occurrence {index} has no hexadecimal word")
        try:
            word = int(raw_word, 16)
        except ValueError as exc:
            raise OpenRiscCorpusError(
                f"occurrence {index} has an invalid hexadecimal word"
            ) from exc
        if not 0 <= word <= 0xFFFF_FFFF:
            raise OpenRiscCorpusError(
                f"occurrence {index} is not an aligned 32-bit instruction word"
            )
        counts[word] += 1
    return sorted(counts), counts


def analyze_openrisc_corpus(
    ledger: Mapping[str, Any], artifact_paths: Iterable[str | Path]
) -> dict[str, Any]:
    """Return exact aligned and unaligned byte matches for unresolved words."""

    words, source_counts = _unresolved_words(ledger)
    sources = [Path(path).resolve() for path in artifact_paths]
    if not sources:
        raise OpenRiscCorpusError("no corpus artifacts supplied")
    names = [path.name for path in sources]
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate_names:
        raise OpenRiscCorpusError(
            f"duplicate artifact name: {', '.join(duplicate_names)}"
        )

    artifacts: list[dict[str, Any]] = []
    matches_by_word: dict[int, list[dict[str, str]]] = {word: [] for word in words}
    for source in sorted(sources, key=lambda path: path.name):
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise OpenRiscCorpusError(f"cannot read corpus artifact {source}: {exc}") from exc
        family_counts = Counter(
            int.from_bytes(data[offset : offset + 4], "little") >> 26
            for offset in range(0, len(data) - 3, 4)
        )
        artifacts.append(
            {
                "name": source.name,
                "path": str(source),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "aligned_little_endian_family_word_counts": {
                    f"0x{opcode:02x}": family_counts[opcode]
                    for opcode in _TRACKED_OPCODES
                },
            }
        )
        for word in words:
            encodings = (("little", word.to_bytes(4, "little")),)
            if word.to_bytes(4, "big") != word.to_bytes(4, "little"):
                encodings += (("big", word.to_bytes(4, "big")),)
            for byte_order, needle in encodings:
                for offset in _find_all(data, needle):
                    matches_by_word[word].append(
                        {
                            "artifact": source.name,
                            "byte_order": byte_order,
                            "alignment": "aligned_4" if offset % 4 == 0 else "unaligned",
                            "offset": _hex32(offset),
                        }
                    )

    word_records: list[dict[str, Any]] = []
    all_matches: list[dict[str, str]] = []
    for word in words:
        matches = sorted(
            matches_by_word[word],
            key=lambda item: (
                item["artifact"],
                int(item["offset"], 16),
                0 if item["byte_order"] == "little" else 1,
            ),
        )
        all_matches.extend(matches)
        word_records.append(
            {
                "word": _hex32(word),
                "opcode": f"0x{word >> 26:02x}",
                "source_occurrence_count": source_counts[word],
                "matches": matches,
            }
        )

    aligned_little = sum(
        match["alignment"] == "aligned_4" and match["byte_order"] == "little"
        for match in all_matches
    )
    unaligned_little = sum(
        match["alignment"] == "unaligned" and match["byte_order"] == "little"
        for match in all_matches
    )
    aligned_big = sum(
        match["alignment"] == "aligned_4" and match["byte_order"] == "big"
        for match in all_matches
    )
    return {
        "schema": "verkeye.compat.openrisc-cross-corpus.v1",
        "source_ledger_schema": ledger["schema"],
        "artifacts": artifacts,
        "summary": {
            "artifact_count": len(artifacts),
            "unique_unresolved_word_count": len(words),
            "aligned_little_endian_match_count": aligned_little,
            "unaligned_little_endian_match_count": unaligned_little,
            "aligned_big_endian_match_count": aligned_big,
            "semantics_resolved": False,
            "inference_execution_supported": False,
        },
        "words": word_records,
        "gate": {
            "status": "blocked",
            "reason_codes": [
                "CROSS_CORPUS_MATCHES_DO_NOT_DEFINE_INSTRUCTION_SEMANTICS"
            ],
        },
        "claim_scope": (
            "The recovered ORC images are mixed code/data blobs. Exact byte matches "
            "are separated by byte order and four-byte alignment. Even an aligned "
            "match does not establish that the bytes are executable, share the same "
            "ISA, or implement the same state transition. No match is interpreted as "
            "instruction semantics."
        ),
    }
