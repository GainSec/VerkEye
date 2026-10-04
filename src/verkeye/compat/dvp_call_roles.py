"""Prove named DVP wrapper roles in the recovered CV22 firmware.

The firmware embeds diagnostic strings that name three DVP wrappers.  This
module binds those strings to exact function ranges, incoming calls, direct
outgoing calls, and implementation-defined instruction words.  Names and call
relationships constrain reverse engineering; they do not define the custom
instructions' state transitions.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ambarella_custom import classify_unsupported_word
from .openrisc import OpenRiscFirmwareError, analyze_openrisc_firmware


class DvpCallRoleError(ValueError):
    """The recovered firmware no longer matches the pinned DVP call roles."""


_RUNTIME_BASE = 0x0040_0000
_RETURN_WORD = 0x4400_4800
_CUSTOM_OPCODES = {0x07, 0x10, 0x1C, 0x1F}


@dataclass(frozen=True, slots=True)
class _Wrapper:
    name: str
    diagnostic_offset: int
    function_start: int
    function_end: int

    @property
    def diagnostic(self) -> bytes:
        return f"{self.name}: Size should be [%u] bytes aligned.\n".encode("ascii")


_WRAPPERS = (
    _Wrapper("Dvp_xor_dram", 0x10F6B, 0x10FA0, 0x110BC),
    _Wrapper("Dvp_xor_vmem", 0x110BB, 0x110F0, 0x111C8),
    _Wrapper("Dvp_clear_vmem", 0x111C7, 0x111FC, 0x112B0),
)


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _word(data: bytes, offset: int) -> int:
    try:
        return struct.unpack_from("<I", data, offset)[0]
    except struct.error as exc:
        raise DvpCallRoleError(f"cannot read word at {_hex32(offset)}") from exc


def _jal_target(word: int, runtime_address: int) -> int | None:
    if word >> 26 != 0x01:
        return None
    displacement = word & 0x03FF_FFFF
    if displacement & 0x0200_0000:
        displacement -= 0x0400_0000
    return (runtime_address + (displacement << 2)) & 0xFFFF_FFFF


def _call_record(offset: int, target: int | None = None) -> dict[str, str]:
    record = {
        "file_offset": _hex32(offset),
        "runtime_address": _hex32(_RUNTIME_BASE + offset),
    }
    if target is not None:
        record["target_runtime_address"] = _hex32(target)
    return record


def analyze_dvp_call_roles(path: str | Path) -> dict[str, Any]:
    """Return exact named-wrapper and call-role evidence."""

    source = Path(path).resolve()
    try:
        firmware = analyze_openrisc_firmware(source)
    except OpenRiscFirmwareError as exc:
        raise DvpCallRoleError(f"firmware identity validation failed: {exc}") from exc
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise DvpCallRoleError(f"cannot read firmware {source}: {exc}") from exc

    for wrapper in _WRAPPERS:
        observed = data[
            wrapper.diagnostic_offset : wrapper.diagnostic_offset
            + len(wrapper.diagnostic)
        ]
        if observed != wrapper.diagnostic:
            raise DvpCallRoleError(
                f"missing diagnostic string for {wrapper.name} at "
                f"{_hex32(wrapper.diagnostic_offset)}"
            )
        if wrapper.function_start % 4 or wrapper.function_end % 4:
            raise DvpCallRoleError(f"unaligned function range for {wrapper.name}")
        if wrapper.function_end > len(data):
            raise DvpCallRoleError(f"truncated function for {wrapper.name}")
        if _word(data, wrapper.function_end - 4) != _RETURN_WORD:
            raise DvpCallRoleError(f"return boundary changed for {wrapper.name}")

    aligned_words = [
        (offset, _word(data, offset)) for offset in range(0, len(data) - 3, 4)
    ]
    wrappers: list[dict[str, Any]] = []
    incoming_total = 0
    custom_total = 0
    for wrapper in _WRAPPERS:
        runtime_start = _RUNTIME_BASE + wrapper.function_start
        incoming = [
            _call_record(offset)
            for offset, word in aligned_words
            if _jal_target(word, _RUNTIME_BASE + offset) == runtime_start
        ]
        direct_calls: list[dict[str, str]] = []
        custom: list[dict[str, object]] = []
        for offset in range(wrapper.function_start, wrapper.function_end, 4):
            word = _word(data, offset)
            target = _jal_target(word, _RUNTIME_BASE + offset)
            if target is not None:
                direct_calls.append(_call_record(offset, target))
            if word >> 26 in _CUSTOM_OPCODES:
                custom.append(classify_unsupported_word(word, offset))
        if not incoming:
            raise DvpCallRoleError(f"no incoming direct call for {wrapper.name}")
        incoming_total += len(incoming)
        custom_total += len(custom)
        wrappers.append(
            {
                "name": wrapper.name,
                "diagnostic_string": {
                    "file_offset": _hex32(wrapper.diagnostic_offset),
                    "runtime_address": _hex32(
                        _RUNTIME_BASE + wrapper.diagnostic_offset
                    ),
                    "text": wrapper.diagnostic.decode("ascii"),
                },
                "function": {
                    "file_offset_start": _hex32(wrapper.function_start),
                    "file_offset_end_exclusive": _hex32(wrapper.function_end),
                    "runtime_address_start": _hex32(runtime_start),
                },
                "incoming_calls": incoming,
                "direct_calls": direct_calls,
                "custom_instructions": custom,
            }
        )

    return {
        "schema": "verkeye.compat.dvp-call-roles.v1",
        "artifact": firmware["artifact"],
        "summary": {
            "named_wrapper_count": len(wrappers),
            "incoming_call_count": incoming_total,
            "custom_instruction_count": custom_total,
            "semantics_resolved": False,
            "inference_execution_supported": False,
        },
        "wrappers": wrappers,
        "gate": {
            "status": "blocked",
            "reason_codes": [
                "NAMED_WRAPPER_DOES_NOT_DEFINE_CUSTOM_INSTRUCTION_SEMANTICS"
            ],
        },
        "claim_scope": (
            "Embedded diagnostic strings prove the names of three DVP wrappers. "
            "Aligned l.jal encodings prove their incoming and outgoing direct-call "
            "relationships, and exact firmware words prove each implementation-"
            "defined instruction sequence. Function names constrain operation roles "
            "but do not define operands, hidden DVP state, synchronization, or exact "
            "state transitions."
        ),
    }
