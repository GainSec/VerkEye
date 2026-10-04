"""Reproducible QEMU/OpenRISC compatibility evidence for Ambarella CV22.

This module validates the isolated no-delay CPU patch and the first bounded
execution trace produced with it.  It deliberately stops at the first
unimplemented Ambarella instruction; it does not assign that instruction a
state transition or claim that inference is executable.
"""

from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from typing import Any

from .ambarella_custom import classify_unsupported_word
from .openrisc import analyze_openrisc_firmware


class QemuCv22Error(ValueError):
    """The QEMU patch or trace does not match the pinned CV22 experiment."""


_QEMU_REPOSITORY = "https://gitlab.com/qemu-project/qemu.git"
_QEMU_COMMIT = "f7ada39edacaa5c26b30e98b94017b0b2ccbcf94"
_PATCH_SHA256 = "0fc223db6330f5e54b6c90dfd1585a42ddf3f2eaff8803c2e0d706965d29b6c5"
_TRACE_SHA256 = "9b8a11c19b4d7b364bd6a5a2ccd2a275358b23543a1d0ea044aac0afb5ea114e"
_BOUNDARY_OFFSET = 0x0001_0D30
_BOUNDARY_PC = 0x0041_0D30
_BOUNDARY_WORD = 0x4000_0020
_EXCEPTION_PC = 0x0000_0700
_RUNTIME_BASE = 0x0040_0000
_INSTRUCTION = re.compile(
    r"^0x(?P<address>[0-9a-fA-F]{8}):\s+(?P<disassembly>.+?)\s*$",
    re.MULTILINE,
)
_PC = re.compile(r"^PC=(?P<pc>[0-9a-fA-F]{8})$", re.MULTILINE)
_REGISTER = re.compile(r"R(?P<index>\d{2})=(?P<value>[0-9a-fA-F]{8})")


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _read(path: str | Path, role: str) -> bytes:
    source = Path(path).resolve()
    try:
        return source.read_bytes()
    except OSError as exc:
        raise QemuCv22Error(f"cannot read {role} {source}: {exc}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_trace(raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise QemuCv22Error("trace is not ASCII") from exc

    blocks: list[dict[str, Any]] = []
    for part in text.split("----------------\n"):
        if not part.startswith("IN:"):
            continue
        instructions = [
            {
                "address": int(match.group("address"), 16),
                "disassembly": match.group("disassembly").rstrip(),
            }
            for match in _INSTRUCTION.finditer(part)
        ]
        pc_match = _PC.search(part)
        if not instructions or pc_match is None:
            raise QemuCv22Error("trace block lacks instructions or PC state")
        pc = int(pc_match.group("pc"), 16)
        if instructions[0]["address"] != pc:
            raise QemuCv22Error(
                f"trace block starts at {_hex32(instructions[0]['address'])}, "
                f"but register state reports PC {_hex32(pc)}"
            )
        register_text = part[pc_match.end() :]
        next_pc = _PC.search(register_text)
        if next_pc is not None:
            register_text = register_text[: next_pc.start()]
        registers = {
            int(match.group("index")): int(match.group("value"), 16)
            for match in _REGISTER.finditer(register_text)
        }
        if set(registers) != set(range(32)):
            raise QemuCv22Error("trace block lacks complete register state")
        blocks.append(
            {"pc": pc, "instructions": instructions, "registers": registers}
        )
    if not blocks:
        raise QemuCv22Error("trace contains no instruction blocks")
    return blocks


def _instruction(block: dict[str, Any], address: int) -> dict[str, Any]:
    matches = [
        item for item in block["instructions"] if item["address"] == address
    ]
    if len(matches) != 1:
        raise QemuCv22Error(
            f"expected one instruction at {_hex32(address)}, observed {len(matches)}"
        )
    return matches[0]


def _block_index(blocks: list[dict[str, Any]], pc: int) -> int:
    matches = [index for index, block in enumerate(blocks) if block["pc"] == pc]
    if len(matches) != 1:
        raise QemuCv22Error(
            f"expected one trace block at {_hex32(pc)}, observed {len(matches)}"
        )
    return matches[0]


def _validate_patch(raw: bytes) -> list[str]:
    if _sha256(raw) != _PATCH_SHA256:
        raise QemuCv22Error("QEMU CV22 patch sha256 mismatch")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise QemuCv22Error("QEMU CV22 patch is not ASCII") from exc
    touched = re.findall(r"^diff --git a/(\S+) b/(\S+)$", text, re.MULTILINE)
    if any(left != right for left, right in touched):
        raise QemuCv22Error("QEMU CV22 patch renames an unexpected file")
    files = [left for left, _ in touched]
    expected = [
        "target/or1k/cpu.c",
        "target/or1k/cpu.h",
        "target/or1k/disas.c",
        "target/or1k/helper.h",
        "target/or1k/insns.decode",
        "target/or1k/sys_helper.c",
        "target/or1k/translate.c",
    ]
    if files != expected:
        raise QemuCv22Error(f"QEMU CV22 patch touched files changed: {files}")
    for required in (
        'DEFINE_OPENRISC_CPU_TYPE("ambarella-cv22", ambarella_cv22_initfn)',
        "cpu->env.cpucfgr |= CPUCFGR_ND;",
        "return !(dc->cpucfgr & CPUCFGR_ND);",
        "dc->base.is_jmp = DISAS_JUMP;",
        "(has_delay_slot(dc) ? 8 : 4)",
        "cpu->env.cv22_extensions = true;",
        "Cv22DvpRegister cv22_dvp_registers[CV22_DVP_REGISTER_COUNT];",
        "cv22_dvp_write 000111 00000 a:5 b:5 00000000010",
        "cv22_dvp_read  000111 00000 a:5 00000 00000000110",
        "if (!env->cv22_extensions || (address & 3))",
        "if (env->cv22_extensions && spr == 0xc060)",
    ):
        if required not in text:
            raise QemuCv22Error(f"QEMU CV22 patch lacks required change: {required}")
    return files


def analyze_cv22_qemu(
    firmware_path: str | Path,
    trace_path: str | Path,
    patch_path: str | Path,
) -> dict[str, Any]:
    """Validate the CV22 no-delay patch and its first execution boundary."""

    firmware = analyze_openrisc_firmware(firmware_path)
    firmware_raw = _read(firmware_path, "firmware")
    patch_raw = _read(patch_path, "patch")
    touched_files = _validate_patch(patch_raw)
    trace_raw = _read(trace_path, "trace")
    blocks = _parse_trace(trace_raw)

    word = struct.unpack_from("<I", firmware_raw, _BOUNDARY_OFFSET)[0]
    if word != _BOUNDARY_WORD:
        raise QemuCv22Error(
            f"firmware word at {_hex32(_BOUNDARY_OFFSET)} changed: {_hex32(word)}"
        )

    first_call_index = _block_index(blocks, 0x0040_00FC)
    first_call = _instruction(blocks[first_call_index], 0x0040_0104)
    if first_call["disassembly"] != "l.jal     17925":
        raise QemuCv22Error("first startup call encoding changed")
    first_target = blocks[first_call_index + 1]
    if first_target["pc"] != 0x0041_1918:
        raise QemuCv22Error("first startup call target changed")
    if first_target["registers"][9] != 0x0040_0108:
        raise QemuCv22Error("first startup call did not create a +4 return link")
    _instruction(first_target, 0x0041_1920)
    returned = blocks[first_call_index + 2]
    if returned["pc"] != 0x0040_0108:
        raise QemuCv22Error("first startup call did not return to the next word")

    second_call = _instruction(returned, 0x0040_0108)
    if second_call["disassembly"] != "l.jal     17139":
        raise QemuCv22Error("second startup call encoding changed")
    second_target = blocks[first_call_index + 3]
    if second_target["pc"] != 0x0041_0CD4:
        raise QemuCv22Error("second startup call target changed")
    if second_target["registers"][9] != 0x0040_010C:
        raise QemuCv22Error("second startup call did not create a +4 return link")

    unsupported = _instruction(second_target, _BOUNDARY_PC)
    expected_disassembly = f".long     {_hex32(_BOUNDARY_WORD)}"
    if unsupported["disassembly"] != expected_disassembly:
        raise QemuCv22Error("first unsupported instruction does not match firmware")
    following_index = first_call_index + 4
    if following_index >= len(blocks) or blocks[following_index]["pc"] != _EXCEPTION_PC:
        raise QemuCv22Error("unsupported instruction did not reach exception vector")
    if _sha256(trace_raw) != _TRACE_SHA256:
        raise QemuCv22Error("CV22 QEMU trace sha256 mismatch")

    instruction = classify_unsupported_word(word, _BOUNDARY_OFFSET)
    return {
        "schema": "verkeye.compat.qemu-cv22.v1",
        "firmware": firmware["artifact"],
        "qemu": {
            "repository": _QEMU_REPOSITORY,
            "commit": _QEMU_COMMIT,
            "machine": "or1k-sim",
            "cpu": "ambarella-cv22",
            "ram_bytes": 4_294_967_296,
            "cpu_configuration": {"CPUCFGR.ND": 1},
            "branch_semantics": "no_delay_slot",
            "implementation_defined_extensions": {
                "opcode_0x07_function_0x002": "dvp_register_write",
                "opcode_0x07_function_0x006": (
                    "dvp_register_read_to_spr_0xc060"
                ),
                "unknown_read_policy": "illegal_instruction",
                "all_other_extensions": "illegal_instruction",
            },
        },
        "architecture_reference": {
            "repository": "https://github.com/openrisc/doc.git",
            "commit": "3c839e1ed231ebeb32a17a900eaa1168414b29ee",
            "manual": "OpenRISC Architecture Manual 1.4",
            "field": "CPUCFGR.ND",
            "nd_zero": "execute_delay_slot",
            "nd_one": "do_not_execute_delay_slot",
            "link_when_nd_one": "jump_instruction_address_plus_4",
        },
        "patch": {
            "path_role": "qemu_or1k_cv22_patch",
            "size": len(patch_raw),
            "sha256": _sha256(patch_raw),
            "touched_files": touched_files,
        },
        "trace": {
            "path_role": "bounded_qemu_cv22_no_delay_boundary_trace",
            "size": len(trace_raw),
            "sha256": _sha256(trace_raw),
            "input_image_sha256": (
                "af654c74940b0842f86aca6eaa9e844a9492f44bae3d9c9a03c3bff82e780494"
            ),
            "input_image_note": (
                "The diagnostic image changes only 0x004120d8, which was not "
                "reached before this earlier boundary. It remains non-faithful."
            ),
        },
        "execution": {
            "startup_call": {
                "call_runtime_address": "0x00400104",
                "target_runtime_address": "0x00411918",
                "return_link": "0x00400108",
                "returned_to": "0x00400108",
                "delay_slot_executed": False,
            },
            "next_startup_call": {
                "call_runtime_address": "0x00400108",
                "target_runtime_address": "0x00410cd4",
                "return_link": "0x0040010c",
            },
            "first_unsupported_instruction": instruction,
            "next_pc": _hex32(_EXCEPTION_PC),
            "qemu_result": "illegal_instruction_exception",
        },
        "fidelity_gate": "blocked_on_implementation_defined_instruction_semantics",
        "claim_scope": (
            "The isolated CPU model implements the architecture-defined CV22 "
            "no-delay branch mode and executes sequential startup calls. Execution "
            "then stops at the first dynamically reached unallocated CV22 opcode. "
            "No implementation-defined state transition or inference result is "
            "inferred."
        ),
    }
