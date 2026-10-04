"""Hash-pinned static context for trace-reached CV22 OpenRISC words.

This module deliberately stops short of interpreting implementation-defined
instructions.  It proves that an external disassembly matches the exact
firmware bytes, records uninterpreted bit fields, and maps direct ``l.jal``
callers to dynamically observed boundary addresses.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class OpenRiscStaticError(ValueError):
    """Static evidence is malformed or does not match the pinned firmware."""


@dataclass(frozen=True, slots=True)
class Instruction:
    address: int
    raw: bytes
    text: str

    @property
    def word(self) -> int:
        return int.from_bytes(self.raw, "little")


_INSTRUCTION = re.compile(
    r"^\s*(?P<address>[0-9a-fA-F]+):\s+"
    r"(?P<bytes>[0-9a-fA-F]{2}(?:\s+[0-9a-fA-F]{2}){3})\s+"
    r"(?P<text>\S.*)$"
)
_DIRECT_CALL = re.compile(r"^l\.jal\s+0x(?P<target>[0-9a-fA-F]+)$")
_HEX32 = re.compile(r"^0x[0-9a-fA-F]{1,8}$")


def sha256(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise OpenRiscStaticError("artifact content must be bytes")
    return hashlib.sha256(data).hexdigest()


def _artifact(name: str, data: bytes) -> dict[str, object]:
    return {"name": name, "size": len(data), "sha256": sha256(data)}


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _parse_hex32(value: object, *, field: str) -> int:
    if not isinstance(value, str) or _HEX32.fullmatch(value) is None:
        raise OpenRiscStaticError(f"{field} must be a hexadecimal 32-bit value")
    return int(value, 16)


def parse_objdump(raw: bytes) -> tuple[Instruction, ...]:
    """Parse the four-byte instruction rows emitted by GNU objdump."""

    if not isinstance(raw, bytes) or not raw:
        raise OpenRiscStaticError("disassembly must be non-empty bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise OpenRiscStaticError("disassembly is not ASCII") from exc
    instructions: list[Instruction] = []
    seen: set[int] = set()
    for line in text.splitlines():
        match = _INSTRUCTION.fullmatch(line)
        if match is None:
            continue
        address = int(match.group("address"), 16)
        if address % 4:
            raise OpenRiscStaticError(
                f"unaligned instruction address {_hex32(address)}"
            )
        if address in seen:
            raise OpenRiscStaticError(
                f"duplicate instruction address {_hex32(address)}"
            )
        seen.add(address)
        instructions.append(
            Instruction(
                address=address,
                raw=bytes.fromhex(match.group("bytes")),
                text=match.group("text").strip(),
            )
        )
    if not instructions:
        raise OpenRiscStaticError("disassembly contains no four-byte instructions")
    return tuple(sorted(instructions, key=lambda item: item.address))


def _public_instruction(item: Instruction) -> dict[str, object]:
    return {
        "runtime_address": _hex32(item.address),
        "word": _hex32(item.word),
        "disassembly": item.text,
    }


def analyze_openrisc_static(
    firmware_path: str | Path,
    disassembly_path: str | Path,
    boundaries: Mapping[str, Any],
    *,
    runtime_base: int = 0x0040_0000,
    context_instructions: int = 4,
) -> dict[str, Any]:
    """Correlate exact firmware bytes, static contexts, and direct callers."""

    if (
        not isinstance(runtime_base, int)
        or isinstance(runtime_base, bool)
        or not 0 <= runtime_base <= 0xFFFF_FFFF
    ):
        raise OpenRiscStaticError("runtime_base must be a 32-bit integer")
    if (
        not isinstance(context_instructions, int)
        or isinstance(context_instructions, bool)
        or context_instructions < 0
    ):
        raise OpenRiscStaticError("context_instructions must be non-negative")

    firmware_source = Path(firmware_path).resolve()
    disassembly_source = Path(disassembly_path).resolve()
    try:
        firmware = firmware_source.read_bytes()
        raw_disassembly = disassembly_source.read_bytes()
    except OSError as exc:
        raise OpenRiscStaticError(f"cannot read static-analysis input: {exc}") from exc

    instructions = parse_objdump(raw_disassembly)
    by_address = {item.address: item for item in instructions}
    for item in instructions:
        offset = item.address - runtime_base
        if offset < 0 or offset + 4 > len(firmware):
            raise OpenRiscStaticError(
                f"disassembly address {_hex32(item.address)} is outside firmware"
            )
        if firmware[offset : offset + 4] != item.raw:
            raise OpenRiscStaticError(
                f"disassembly at {_hex32(item.address)} differs from firmware"
            )

    raw_boundaries = boundaries.get("dynamically_reached_instructions")
    if not isinstance(raw_boundaries, list) or not raw_boundaries:
        raise OpenRiscStaticError(
            "boundary report lacks dynamically_reached_instructions"
        )

    direct_calls: dict[int, list[Instruction]] = {}
    for item in instructions:
        match = _DIRECT_CALL.fullmatch(item.text)
        if match is not None:
            direct_calls.setdefault(int(match.group("target"), 16), []).append(item)

    callable_entries = sorted(
        target for target in direct_calls if target in by_address
    )
    return_addresses = {
        item.address for item in instructions if item.text == "l.jr r9"
    }

    def enclosing_direct_call_target(address: int) -> int | None:
        """Find the nearest called entry with no intervening return instruction."""

        for candidate in reversed(callable_entries):
            if candidate > address:
                continue
            if not any(candidate <= returned < address for returned in return_addresses):
                return candidate
        return None

    address_to_index = {item.address: index for index, item in enumerate(instructions)}
    output_boundaries: list[dict[str, object]] = []
    observed_call_edges: set[tuple[int, int]] = set()
    observed_addresses: set[int] = set()
    for position, record in enumerate(raw_boundaries):
        if not isinstance(record, Mapping):
            raise OpenRiscStaticError(f"boundary {position} is not an object")
        address = _parse_hex32(
            record.get("runtime_address"), field=f"boundary {position} address"
        )
        expected_word = _parse_hex32(
            record.get("word"), field=f"boundary {position} word"
        )
        if address in observed_addresses:
            raise OpenRiscStaticError(f"duplicate boundary at {_hex32(address)}")
        observed_addresses.add(address)
        item = by_address.get(address)
        if item is None:
            raise OpenRiscStaticError(
                f"boundary {_hex32(address)} is missing from disassembly"
            )
        if item.word != expected_word:
            raise OpenRiscStaticError(
                f"boundary {_hex32(address)} word differs from boundary report"
            )
        offset = address - runtime_base
        index = address_to_index[address]
        start = max(0, index - context_instructions)
        end = min(len(instructions), index + context_instructions + 1)
        enclosing_target = enclosing_direct_call_target(address)
        callers = direct_calls.get(enclosing_target, [])
        if enclosing_target is not None:
            observed_call_edges.update(
                (enclosing_target, caller.address) for caller in callers
            )
        output_boundaries.append(
            {
                "instruction": {
                    "runtime_address": _hex32(address),
                    "file_offset": _hex32(offset),
                    "bytes_little_endian": item.raw.hex(),
                    "word": _hex32(item.word),
                    "disassembly": item.text,
                    "opcode": f"0x{item.word >> 26:02x}",
                    "bit_fields_uninterpreted": {
                        "bits_25_21": (item.word >> 21) & 0x1F,
                        "bits_20_16": (item.word >> 16) & 0x1F,
                        "bits_15_11": (item.word >> 11) & 0x1F,
                        "bits_10_0": f"0x{item.word & 0x7FF:03x}",
                    },
                    "semantics": "unresolved",
                },
                "context": [
                    _public_instruction(context) for context in instructions[start:end]
                ],
                "enclosing_direct_call_target": (
                    _hex32(enclosing_target) if enclosing_target is not None else None
                ),
                "direct_callers": [_public_instruction(caller) for caller in callers],
            }
        )

    return {
        "schema": "verkeye.compat.openrisc-static.v1",
        "artifact": _artifact(firmware_source.name, firmware),
        "disassembly_artifact": _artifact(
            disassembly_source.name, raw_disassembly
        ),
        "runtime_base": _hex32(runtime_base),
        "summary": {
            "boundary_count": len(output_boundaries),
            "direct_call_count": len(observed_call_edges),
            "state_transitions_resolved": False,
            "inference_execution_supported": False,
        },
        "boundaries": output_boundaries,
        "claim_scope": (
            "This report proves byte-exact static context and direct l.jal callers "
            "for the nearest called entry preceding each trace-reached "
            "implementation-defined word without an intervening l.jr r9 return. "
            "The named bit fields are positional slices only, not decoded operands. "
            "No instruction semantics or state transition is inferred."
        ),
    }
