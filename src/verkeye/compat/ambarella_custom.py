"""Inventory trace-reached unsupported words in the CV22 scheduler.

OpenRISC opcode values are the instruction's high six bits.  The architecture
table leaves opcodes 0x07 and 0x10 unallocated and reserves 0x1c and 0x1f as
``l.cust1`` and ``l.cust4``.  This module records exact encodings only; it does
not invent operand fields, emulate a family, or infer a state transition.
"""

from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path
from typing import Any

from .openrisc import analyze_openrisc_firmware


class AmbarellaCustomError(ValueError):
    """The input does not match the bounded implementation-specific evidence."""


_RUNTIME_BASE = 0x0040_0000
_FUNCTIONS = (
    (0x1206C, 0x12100),
    (0x12100, 0x1217C),
    (0x1217C, 0x121D0),
    (0x121D0, 0x12220),
    (0x12220, 0x12274),
    (0x12274, 0x12310),
    (0x12310, 0x123A8),
    (0x123A8, 0x12454),
)
_REGION_START = _FUNCTIONS[0][0]
_REGION_END = _FUNCTIONS[-1][1]
_UNALLOCATED_OPCODES = {0x07, 0x10}
_CUSTOM_OPCODES = {0x1C: "l.cust1", 0x1F: "l.cust4"}
_STATIC_CANDIDATE_OPCODES = _UNALLOCATED_OPCODES
_DYNAMIC_BOUNDARIES = (
    (0x10D30, 0x4000_0020),
    (0x10D48, 0x4000_0000),
    (0x120D8, 0x1C04_4012),
    (0x10D68, 0x70C0_0006),
    (0x12190, 0x1C09_1812),
    (0x12198, 0x1C08_2012),
    (0x121A4, 0x1C06_3812),
    (0x121AC, 0x1C04_0009),
    (0x121B4, 0x1C04_0008),
    (0x121B8, 0x4000_0028),
    (0x0FF60, 0x7C9B_0008),
    (0x0FF68, 0x7C89_0008),
    (0x0FF8C, 0x7D00_5808),
    (0x0FF90, 0x7E00_5808),
    (0x0FF94, 0x7D00_5808),
    (0x0FF98, 0x7E00_5808),
    (0x09F6C, 0x7000_0004),
    (0x10064, 0x7D00_7008),
    (0x10068, 0x7E00_7008),
    (0x100C8, 0x7C84_0008),
    (0x100CC, 0x4000_0020),
    (0x100E0, 0x4000_0000),
    (0x100E8, 0x7E00_9008),
)
_ISA_SOURCE = "https://github.com/openrisc/doc.git"
_ISA_COMMIT = "3c839e1ed231ebeb32a17a900eaa1168414b29ee"


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _read_word(data: bytes, offset: int) -> int:
    try:
        return struct.unpack_from("<I", data, offset)[0]
    except struct.error as exc:
        raise AmbarellaCustomError(
            f"cannot read instruction at {_hex32(offset)}"
        ) from exc


def classify_unsupported_word(word: int, file_offset: int) -> dict[str, object]:
    """Classify only pinned unsupported/custom opcode families."""

    if not 0 <= word <= 0xFFFF_FFFF:
        raise AmbarellaCustomError(f"word is outside unsigned 32-bit range: {word}")
    if file_offset < 0 or file_offset % 4:
        raise AmbarellaCustomError(
            f"file offset must be a non-negative aligned word: {file_offset}"
        )

    common: dict[str, object] = {
        "file_offset": _hex32(file_offset),
        "runtime_address": _hex32(_RUNTIME_BASE + file_offset),
        "word": _hex32(word),
        "opcode": f"0x{word >> 26:02x}",
    }
    opcode = word >> 26
    if opcode in _UNALLOCATED_OPCODES:
        return {
            **common,
            "family": f"unallocated_orbis32_opcode_0x{opcode:02x}",
            "payload_bits": f"0x{word & 0x03FF_FFFF:07x}",
            "semantics": "unresolved",
        }
    if opcode in _CUSTOM_OPCODES:
        return {
            **common,
            "family": _CUSTOM_OPCODES[opcode],
            "payload_bits": f"0x{word & 0x03FF_FFFF:07x}",
            "semantics": "unresolved",
        }
    raise AmbarellaCustomError(
        f"word {_hex32(word)} at {_hex32(file_offset)} is not a pinned unsupported word"
    )


def analyze_ambarella_custom(path: str | Path) -> dict[str, Any]:
    """Inventory exact custom words in the bounded scheduler helper region."""

    source = Path(path).resolve()
    firmware = analyze_openrisc_firmware(source)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise AmbarellaCustomError(f"cannot read firmware {source}: {exc}") from exc

    instructions: list[dict[str, object]] = []
    functions: list[dict[str, object]] = []
    for start, end in _FUNCTIONS:
        first_index = len(instructions)
        for offset in range(start, end, 4):
            word = _read_word(data, offset)
            if word >> 26 in _STATIC_CANDIDATE_OPCODES:
                instructions.append(classify_unsupported_word(word, offset))
        functions.append(
            {
                "file_offset_start": _hex32(start),
                "file_offset_end_exclusive": _hex32(end),
                "runtime_address_start": _hex32(_RUNTIME_BASE + start),
                "vendor_instruction_count": len(instructions) - first_index,
            }
        )

    families = Counter(item["family"] for item in instructions)
    low_function_bits = Counter(
        f"0x{int(item['word'], 16) & 0x7FF:03x}"
        for item in instructions
        if item["opcode"] == "0x07"
    )
    dynamically_reached: list[dict[str, object]] = []
    for offset, expected_word in _DYNAMIC_BOUNDARIES:
        observed_word = _read_word(data, offset)
        if observed_word != expected_word:
            raise AmbarellaCustomError(
                f"dynamic boundary at {_hex32(offset)} changed from "
                f"{_hex32(expected_word)} to {_hex32(observed_word)}"
            )
        dynamically_reached.append(classify_unsupported_word(observed_word, offset))
    return {
        "schema": "verkeye.compat.ambarella-custom.v2",
        "artifact": firmware["artifact"],
        "isa_reference": {
            "source": _ISA_SOURCE,
            "commit": _ISA_COMMIT,
            "manual": "OpenRISC Architecture Manual 1.4-0",
            "opcode_table": {
                "0x07": "unallocated",
                "0x10": "unallocated",
                "0x1c": "l.cust1_reserved_for_custom_instructions",
                "0x1f": "l.cust4_reserved_for_custom_instructions",
            },
        },
        "region": {
            "file_offset_start": _hex32(_REGION_START),
            "file_offset_end_exclusive": _hex32(_REGION_END),
            "runtime_address_start": _hex32(_RUNTIME_BASE + _REGION_START),
            "runtime_address_end_exclusive": _hex32(_RUNTIME_BASE + _REGION_END),
            "size": _REGION_END - _REGION_START,
            "function_count": len(_FUNCTIONS),
        },
        "summary": {
            "instruction_candidate_count": len(instructions),
            "unallocated_opcode_0x07_count": families[
                "unallocated_orbis32_opcode_0x07"
            ],
            "unallocated_opcode_0x10_count": families[
                "unallocated_orbis32_opcode_0x10"
            ],
            "dynamically_reached_instruction_count": len(dynamically_reached),
            "dynamically_reached_family_counts": dict(
                sorted(Counter(item["family"] for item in dynamically_reached).items())
            ),
            "state_transitions_resolved": False,
            "inference_execution_supported": False,
        },
        "opcode_0x07_low_function_bits": dict(sorted(low_function_bits.items())),
        "functions": functions,
        "instructions": instructions,
        "dynamically_reached_instructions": dynamically_reached,
        "claim_scope": (
            "This report proves exact encodings in a bounded helper region and "
            "23 separately trace-reached control-flow boundaries. OpenRISC leaves "
            "opcodes 0x07 and 0x10 unallocated and reserves 0x1c/l.cust1 and "
            "0x1f/l.cust4 for implementation-defined instructions. No operand "
            "layout or state transition is inferred."
        ),
    }
