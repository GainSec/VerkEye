"""Fail-closed OpenRISC evidence for the exact recovered CV22 firmware.

This module deliberately implements only the standard OpenRISC encodings and
vendor-extension boundary that are proven by the pinned firmware and execution
trace.  It is an evidence analyzer, not an Ambarella instruction emulator.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any


class OpenRiscFirmwareError(ValueError):
    """The input does not match the exact firmware or its proven semantics."""


_SIZE = 101_332
_SHA256 = "889c32b8b599454b1a98dfac4e6f9ee7f25abe8f02e21030646e697b07aa45f3"
_RUNTIME_BASE = 0x0040_0000
_NOP = 0x1500_0000

_VECTORS = (
    (0x00, 0x64),
    (0x08, 0x158),
    (0x10, 0x240),
)

_STRING_REFERENCES = (
    (0x1011C, 0x10120, 6, 0xFC14, "CMD 0x%08x is not supported."),
    (0x10178, 0x1017C, 6, 0xFC32, "STOP_CMD is found."),
    (0x10248, 0x1024C, 6, 0xFC46, "DAG_RUN_CMD is found."),
    (0x10760, 0x10764, 6, 0xFDA6, "Cavalry ucode is started."),
    (0x10B64, 0x10B68, 6, 0x109E3, "Cavalry Scheduler Started."),
)

_FIRST_EXTENSION_OFFSET = 0x120D8
_FIRST_EXTENSION_WORD = 0x1C04_4012
_NEXT_EXTENSION_OFFSET = 0x12118
_NEXT_EXTENSION_WORD = 0x1C13_A812


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _word(data: bytes, offset: int) -> int:
    try:
        return struct.unpack_from("<I", data, offset)[0]
    except struct.error as exc:
        raise OpenRiscFirmwareError(
            f"cannot read 32-bit instruction at {_hex32(offset)}"
        ) from exc


def word_swap_32(data: bytes) -> bytes:
    """Convert little-endian instruction words for big-endian QEMU OR1K.

    The byte length must be word-aligned.  Silently retaining a partial word
    would create a transformed image with undefined instruction/data meaning.
    """

    if len(data) % 4:
        raise OpenRiscFirmwareError(
            f"firmware size {len(data)} is not aligned to 32-bit words"
        )
    return b"".join(data[index : index + 4][::-1] for index in range(0, len(data), 4))


def _read_exact(path: str | Path) -> bytes:
    source = Path(path).resolve()
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise OpenRiscFirmwareError(f"cannot read firmware {source}: {exc}") from exc
    if len(data) != _SIZE:
        raise OpenRiscFirmwareError(
            f"firmware size mismatch: expected {_SIZE}, observed {len(data)}"
        )
    digest = _sha256(data)
    if digest != _SHA256:
        raise OpenRiscFirmwareError(
            f"firmware sha256 mismatch: expected {_SHA256}, observed {digest}"
        )
    return data


def _decode_jump(data: bytes, offset: int, expected_target: int) -> dict[str, object]:
    instruction = _word(data, offset)
    if instruction >> 26 != 0:
        raise OpenRiscFirmwareError(
            f"entry vector at {_hex32(offset)} is not OpenRISC l.j"
        )
    displacement = instruction & 0x03FF_FFFF
    if displacement & 0x0200_0000:
        displacement -= 0x0400_0000
    target = offset + displacement * 4
    if target != expected_target:
        raise OpenRiscFirmwareError(
            f"entry vector at {_hex32(offset)} targets {_hex32(target)}, "
            f"expected {_hex32(expected_target)}"
        )
    delay_offset = offset + 4
    delay_word = _word(data, delay_offset)
    if delay_word != _NOP:
        raise OpenRiscFirmwareError(
            f"entry vector delay slot at {_hex32(delay_offset)} is not l.nop"
        )
    return {
        "offset": _hex32(offset),
        "word": _hex32(instruction),
        "mnemonic": "l.j",
        "target_offset": _hex32(target),
        "target_runtime_address": _hex32(_RUNTIME_BASE + target),
        "delay_slot_offset": _hex32(delay_offset),
        "delay_slot_word": _hex32(delay_word),
        "delay_slot_mnemonic": "l.nop",
    }


def _decode_string_reference(
    data: bytes,
    movhi_offset: int,
    ori_offset: int,
    expected_register: int,
    expected_target: int,
    text: str,
) -> dict[str, object]:
    movhi = _word(data, movhi_offset)
    ori = _word(data, ori_offset)
    movhi_opcode = movhi >> 26
    ori_opcode = ori >> 26
    movhi_register = (movhi >> 21) & 0x1F
    ori_destination = (ori >> 21) & 0x1F
    ori_source = (ori >> 16) & 0x1F
    if (
        movhi_opcode != 0x06
        or ori_opcode != 0x2A
        or movhi_register != expected_register
        or ori_destination != expected_register
        or ori_source != expected_register
    ):
        raise OpenRiscFirmwareError(
            "absolute string reference does not match l.movhi/l.ori encoding at "
            f"{_hex32(movhi_offset)}/{_hex32(ori_offset)}"
        )
    address = ((movhi & 0xFFFF) << 16) | (ori & 0xFFFF)
    target = address - _RUNTIME_BASE
    if target != expected_target:
        raise OpenRiscFirmwareError(
            f"string reference resolves to {_hex32(address)}, expected "
            f"{_hex32(_RUNTIME_BASE + expected_target)}"
        )
    encoded = text.encode("ascii")
    if data[target : target + len(encoded)] != encoded:
        raise OpenRiscFirmwareError(
            f"string reference target {_hex32(target)} does not contain {text!r}"
        )
    return {
        "movhi_offset": _hex32(movhi_offset),
        "ori_offset": _hex32(ori_offset),
        "register": expected_register,
        "target_offset": _hex32(target),
        "target_runtime_address": _hex32(address),
        "text": text,
    }


def analyze_openrisc_firmware(path: str | Path) -> dict[str, Any]:
    """Analyze the exact recovered scheduler with bounded, reproducible claims."""

    data = _read_exact(path)
    vectors = [_decode_jump(data, offset, target) for offset, target in _VECTORS]
    references = [
        _decode_string_reference(data, *reference)
        for reference in _STRING_REFERENCES
    ]

    first_word = _word(data, _FIRST_EXTENSION_OFFSET)
    next_word = _word(data, _NEXT_EXTENSION_OFFSET)
    if first_word != _FIRST_EXTENSION_WORD or next_word != _NEXT_EXTENSION_WORD:
        raise OpenRiscFirmwareError("pinned Ambarella extension words do not match")

    transformed = word_swap_32(data)
    return {
        "schema": "verkeye.compat.openrisc-firmware.v1",
        "artifact": {
            "path_role": "cavalry_firmware",
            "size": len(data),
            "sha256": _sha256(data),
        },
        "classification": {
            "architecture": "OpenRISC 1000",
            "instruction_byte_order": "little",
            "runtime_base": _hex32(_RUNTIME_BASE),
            "vendor_extensions_present": True,
            "inference_execution_supported": False,
            "unresolved_boundary": (
                "Ambarella OpenRISC extension semantics and accelerator state "
                "transitions"
            ),
        },
        "entry_vectors": vectors,
        "absolute_string_references": references,
        "qemu_transform": {
            "operation": "reverse each 32-bit word",
            "size": len(transformed),
            "sha256": _sha256(transformed),
            "reason": (
                "qemu-system-or1k executes big-endian instruction words while the "
                "recovered image stores OpenRISC words little-endian"
            ),
        },
        "qemu_boundary": {
            "machine": "or1k-sim",
            "cpu": "or1200",
            "load_address": _hex32(_RUNTIME_BASE),
            "word_swap_required": True,
            "first_unsupported_file_offset": _hex32(_FIRST_EXTENSION_OFFSET),
            "first_unsupported_runtime_address": _hex32(
                _RUNTIME_BASE + _FIRST_EXTENSION_OFFSET
            ),
            "first_unsupported_word": _hex32(first_word),
            "next_observed_extension": {
                "file_offset": _hex32(_NEXT_EXTENSION_OFFSET),
                "runtime_address": _hex32(_RUNTIME_BASE + _NEXT_EXTENSION_OFFSET),
                "word": _hex32(next_word),
            },
            "claim_scope": (
                "QEMU executes the standard OpenRISC bootstrap through 0x004120d4; "
                "execution traps on the instruction at 0x004120d8"
            ),
        },
    }
