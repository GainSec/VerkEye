"""Bounded execution probe for the two evidenced CV22 DVP operations."""

from __future__ import annotations

import hashlib
import re
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class Cv22DvpProbeError(ValueError):
    """The probe or trace violates the exact bounded test contract."""


_LOAD_ADDRESS = 0x0040_0000
_RESULT_PC = 0x0040_001C
_DVP_ADDRESS = 0x0012_000C
_EXPECTED_RESULT = 0xA5AF_E5E7
_EXCEPTION_PC = 0x0000_0700
_WORDS = (
    0x1880_0012,
    0xA884_000C,
    0x18A0_A5AF,
    0xA8A5_E5E7,
    0x1C04_2802,
    0x1C04_0006,
    0xB4C0_C060,
    0x0000_0000,
)
_REGISTER = re.compile(r"R(?P<index>\d{2})=(?P<value>[0-9a-fA-F]{8})")
_PC = re.compile(r"^PC=(?P<value>[0-9a-fA-F]{8})$", re.MULTILINE)
_INSTRUCTION_ADDRESS = re.compile(
    r"^0x(?P<value>[0-9a-fA-F]{8}):", re.MULTILINE
)


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _artifact(raw: bytes) -> dict[str, object]:
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@dataclass(frozen=True, slots=True)
class Cv22DvpProbe:
    words: tuple[int, ...]
    image: bytes
    load_address: int
    result_pc: int
    expected_result: int


@dataclass(frozen=True, slots=True)
class Cv22IllegalProbe:
    words: tuple[int, ...]
    image: bytes
    load_address: int
    offending_pc: int


def build_cv22_dvp_probe() -> Cv22DvpProbe:
    """Return the exact big-endian raw image used for write/read validation."""

    return Cv22DvpProbe(
        words=_WORDS,
        image=b"".join(struct.pack(">I", word) for word in _WORDS),
        load_address=_LOAD_ADDRESS,
        result_pc=_RESULT_PC,
        expected_result=_EXPECTED_RESULT,
    )


def build_cv22_dvp_unknown_read_probe() -> Cv22IllegalProbe:
    """Return a read-before-write probe that must trap on the CV22 profile."""

    words = (0x1880_0012, 0xA884_000C, 0x1C04_0006, 0x0000_0000)
    return Cv22IllegalProbe(
        words=words,
        image=b"".join(struct.pack(">I", word) for word in words),
        load_address=_LOAD_ADDRESS,
        offending_pc=0x0040_0008,
    )


def build_cv22_dvp_probe_command(
    qemu: str | Path,
    image: str | Path,
    trace: str | Path,
) -> tuple[str, ...]:
    """Build the bounded system-emulation command for this exact probe."""

    return (
        str(qemu),
        "-M",
        "or1k-sim",
        "-cpu",
        "ambarella-cv22",
        "-m",
        "16M",
        "-nographic",
        "-device",
        f"loader,file={image},addr=0x400000,cpu-num=0,force-raw=on",
        "-d",
        "in_asm,cpu",
        "-D",
        str(trace),
    )


def analyze_cv22_dvp_probe_trace(raw: bytes) -> dict[str, str]:
    """Require the exact result at the probe's stable loop boundary."""

    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Cv22DvpProbeError("trace is not ASCII") from exc

    blocks = text.split("----------------\n")
    matches: list[dict[int, int]] = []
    for block in blocks:
        pcs = [int(match.group("value"), 16) for match in _PC.finditer(block)]
        if _RESULT_PC not in pcs:
            continue
        registers = {
            int(match.group("index")): int(match.group("value"), 16)
            for match in _REGISTER.finditer(block)
        }
        matches.append(registers)
    if not matches:
        raise Cv22DvpProbeError("trace lacks the exact result boundary")

    for registers in matches:
        if all(index in registers for index in (4, 5, 6)):
            if registers[4] != _DVP_ADDRESS or registers[5] != _EXPECTED_RESULT:
                raise Cv22DvpProbeError("probe inputs changed at result boundary")
            if registers[6] != _EXPECTED_RESULT:
                raise Cv22DvpProbeError("SPR 0xc060 latch result mismatch")
            return {
                "result_pc": _hex32(_RESULT_PC),
                "dvp_address": _hex32(_DVP_ADDRESS),
                "written_value": _hex32(_EXPECTED_RESULT),
                "spr_0xc060_result": _hex32(registers[6]),
                "result": "exact_write_read_match",
            }
    raise Cv22DvpProbeError("result boundary lacks required register state")


def analyze_illegal_instruction_trace(
    raw: bytes, offending_pc: int
) -> dict[str, str]:
    """Require an offending instruction block to transition directly to 0x700."""

    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Cv22DvpProbeError("trace is not ASCII") from exc
    blocks: list[tuple[int, set[int]]] = []
    for block in text.split("----------------\n"):
        match = _PC.search(block)
        if match is not None:
            addresses = {
                int(item.group("value"), 16)
                for item in _INSTRUCTION_ADDRESS.finditer(block)
            }
            blocks.append((int(match.group("value"), 16), addresses))
    matches = [
        index for index, (_, addresses) in enumerate(blocks) if offending_pc in addresses
    ]
    if len(matches) != 1:
        raise Cv22DvpProbeError("trace lacks one exact offending boundary")
    index = matches[0]
    if index + 1 >= len(blocks) or blocks[index + 1][0] != _EXCEPTION_PC:
        raise Cv22DvpProbeError("offending instruction did not directly reach 0x700")
    return {
        "offending_pc": _hex32(offending_pc),
        "exception_pc": _hex32(_EXCEPTION_PC),
        "result": "illegal_instruction_exception",
    }


def run_bounded_qemu_probe(
    command: tuple[str, ...],
    trace_path: str | Path,
    accept: Callable[[bytes], bool],
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    """Run a looping probe until its trace proves the requested terminal state."""

    if timeout_seconds <= 0:
        raise Cv22DvpProbeError("timeout must be positive")
    trace = Path(trace_path)
    trace.unlink(missing_ok=True)
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + timeout_seconds
    termination = "natural_exit"
    accepted = False
    try:
        while time.monotonic() < deadline:
            raw = trace.read_bytes() if trace.is_file() else b""
            if raw and accept(raw):
                accepted = True
                break
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if not accepted:
            if process.poll() is None:
                raise Cv22DvpProbeError("probe timed out before trace acceptance")
            stderr = b"" if process.stderr is None else process.stderr.read()
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise Cv22DvpProbeError(
                f"probe exited before trace acceptance: {detail or process.returncode}"
            )
        if process.poll() is None:
            termination = "terminate"
            process.terminate()
        process.wait(timeout=1.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1.0)
    raw = trace.read_bytes()
    return {
        "accepted": True,
        "termination": termination,
        "trace_size": len(raw),
    }


def analyze_cv22_dvp_evidence(
    *,
    positive_image: str | Path,
    positive_trace: str | Path,
    noncv_trace: str | Path,
    unknown_image: str | Path,
    unknown_trace: str | Path,
    qemu_commit: str,
    patch_sha256: str,
) -> dict[str, object]:
    """Validate and summarize the bounded positive and negative DVP evidence."""

    positive_image_raw = Path(positive_image).read_bytes()
    unknown_image_raw = Path(unknown_image).read_bytes()
    expected_positive = build_cv22_dvp_probe().image
    expected_unknown = build_cv22_dvp_unknown_read_probe().image
    if positive_image_raw != expected_positive:
        raise Cv22DvpProbeError("positive probe image changed")
    if unknown_image_raw != expected_unknown:
        raise Cv22DvpProbeError("unknown-read probe image changed")

    positive_trace_raw = Path(positive_trace).read_bytes()
    noncv_trace_raw = Path(noncv_trace).read_bytes()
    unknown_trace_raw = Path(unknown_trace).read_bytes()
    return {
        "schema": "verkeye.compat.cv22-dvp-evidence.v1",
        "qemu": {"commit": qemu_commit, "patch_sha256": patch_sha256},
        "implementation_scope": {
            "implemented": [
                "opcode_0x07_function_0x002",
                "opcode_0x07_function_0x006",
            ],
            "hardware_defaults_emulated": False,
            "unknown_reads": "illegal_instruction",
            "all_other_custom_operations": "illegal_instruction",
        },
        "artifacts": {
            "positive_image": _artifact(positive_image_raw),
            "positive_trace": _artifact(positive_trace_raw),
            "generic_cpu_negative_trace": _artifact(noncv_trace_raw),
            "unknown_read_image": _artifact(unknown_image_raw),
            "unknown_read_trace": _artifact(unknown_trace_raw),
        },
        "executions": {
            "positive": analyze_cv22_dvp_probe_trace(positive_trace_raw),
            "generic_cpu_negative": analyze_illegal_instruction_trace(
                noncv_trace_raw, 0x0040_0010
            ),
            "unknown_read_negative": analyze_illegal_instruction_trace(
                unknown_trace_raw, 0x0040_0008
            ),
        },
        "claim_scope": (
            "Only the two firmware-proven opcode-0x07 DVP operations are modeled. "
            "The compatibility state is a bounded sparse register map, not a "
            "claim about CV22 hardware defaults or side effects."
        ),
    }
