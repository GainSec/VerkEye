"""Parse the bounded QEMU trace at the first Ambarella OpenRISC boundary."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .ambarella_custom import classify_unsupported_word
from .openrisc import analyze_openrisc_firmware


class OpenRiscTraceError(ValueError):
    """The trace does not prove the pinned execution boundary."""


_BOUNDARY_PC = 0x0041_20D8
_BOUNDARY_OFFSET = 0x0001_20D8
_BOUNDARY_WORD = 0x1C04_4012
_EXCEPTION_PC = 0x0000_0700
_BLOCK = re.compile(
    r"IN:\s*\n"
    r"0x(?P<address>[0-9a-fA-F]{8}):\s+(?P<disassembly>[^\n]+)\n\s*"
    r"PC=(?P<pc>[0-9a-fA-F]{8})\n"
    r"(?P<registers>(?:R\d{2}=[0-9a-fA-F]{8}(?:\s+|\n)){31}"
    r"R\d{2}=[0-9a-fA-F]{8})",
)
_REGISTER = re.compile(r"R(?P<index>\d{2})=(?P<value>[0-9a-fA-F]{8})")


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _read(path: str | Path) -> bytes:
    source = Path(path).resolve()
    try:
        return source.read_bytes()
    except OSError as exc:
        raise OpenRiscTraceError(f"cannot read trace {source}: {exc}") from exc


def _blocks(trace: bytes) -> list[dict[str, Any]]:
    try:
        text = trace.decode("ascii")
    except UnicodeDecodeError as exc:
        raise OpenRiscTraceError("trace is not ASCII") from exc
    blocks: list[dict[str, Any]] = []
    for match in _BLOCK.finditer(text):
        registers = {
            int(item.group("index")): int(item.group("value"), 16)
            for item in _REGISTER.finditer(match.group("registers"))
        }
        if set(registers) != set(range(32)):
            raise OpenRiscTraceError("trace block lacks complete register state")
        address = int(match.group("address"), 16)
        pc = int(match.group("pc"), 16)
        if address != pc:
            raise OpenRiscTraceError(
                f"trace instruction address {_hex32(address)} does not match PC "
                f"{_hex32(pc)}"
            )
        blocks.append(
            {
                "address": address,
                "disassembly": match.group("disassembly").rstrip(),
                "registers": registers,
            }
        )
    if not blocks:
        raise OpenRiscTraceError("trace contains no complete instruction blocks")
    return blocks


def analyze_openrisc_trace(
    firmware_path: str | Path, trace_path: str | Path
) -> dict[str, Any]:
    """Validate and describe the exact first unsupported-instruction trace."""

    firmware = analyze_openrisc_firmware(firmware_path)
    boundary = firmware["qemu_boundary"]
    if int(boundary["first_unsupported_word"], 16) != _BOUNDARY_WORD:
        raise OpenRiscTraceError("firmware boundary instruction word changed")

    raw_trace = _read(trace_path)
    blocks = _blocks(raw_trace)
    positions = [
        index for index, block in enumerate(blocks) if block["address"] == _BOUNDARY_PC
    ]
    if not positions and b"PC=004120d8" in raw_trace:
        raise OpenRiscTraceError("boundary trace lacks complete register state")
    if len(positions) != 1:
        raise OpenRiscTraceError(
            f"expected one boundary execution block, observed {len(positions)}"
        )
    index = positions[0]
    if index == 0 or index + 1 >= len(blocks):
        raise OpenRiscTraceError("boundary block lacks preceding or following state")
    previous, current, following = blocks[index - 1 : index + 2]
    expected_disassembly = f".long     {_hex32(_BOUNDARY_WORD)}"
    if current["disassembly"] != expected_disassembly:
        raise OpenRiscTraceError(
            "boundary instruction word does not match the pinned firmware"
        )
    if following["address"] != _EXCEPTION_PC:
        raise OpenRiscTraceError(
            f"boundary does not transfer to exception vector {_hex32(_EXCEPTION_PC)}"
        )

    instruction = classify_unsupported_word(_BOUNDARY_WORD, _BOUNDARY_OFFSET)
    registers = current["registers"]
    return {
        "schema": "verkeye.compat.openrisc-trace.v2",
        "trace": {
            "path_role": "bounded_qemu_openrisc_boundary_trace",
            "size": len(raw_trace),
            "sha256": hashlib.sha256(raw_trace).hexdigest(),
        },
        "firmware": firmware["artifact"],
        "instruction": instruction,
        "execution": {
            "previous_instruction": {
                "runtime_address": _hex32(previous["address"]),
                "disassembly": previous["disassembly"],
            },
            "qemu_disassembly": current["disassembly"],
            "pc": _hex32(current["address"]),
            "register_state": {
                f"r{register:02d}": _hex32(value)
                for register, value in sorted(registers.items())
            },
            "next_pc": _hex32(following["address"]),
            "qemu_result": "illegal_instruction_exception",
        },
        "claim_scope": (
            "The exact recovered scheduler reaches unallocated ORBIS32 opcode 0x07 "
            "with a complete captured register state; "
            "unmodified qemu-system-or1k transfers control to exception vector "
            "0x00000700. The architecture defines no operand layout or state "
            "transition for this encoding, so neither is inferred."
        ),
    }
