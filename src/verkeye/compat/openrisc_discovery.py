"""Iteratively discover QEMU-trapping words in exact CV22 firmware.

Every discovered word is replaced by ``l.nop 0`` only in the following
iteration.  This is a control-flow inventory mechanism, never an instruction
emulator: every output is marked non-faithful and cannot satisfy inference or
numerical-fidelity gates.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .openrisc import analyze_openrisc_firmware, word_swap_32


class OpenRiscDiscoveryError(ValueError):
    """A discovery trace or derived image violates the pinned contract."""


@dataclass(frozen=True, slots=True)
class DiscoveryTrace:
    """Raw QEMU trace and deterministic runner provenance for one iteration."""

    raw: bytes
    runner: dict[str, object]


TraceProvider = Callable[[int, bytes], DiscoveryTrace]

_RUNTIME_BASE = 0x0040_0000
_EXCEPTION_PC = 0x0000_0700
_NOP = 0x1500_0000
_BLOCK = re.compile(r"(?:^|----------------\n)IN:\s*\n(?P<body>.*?)(?=----------------\nIN:|\Z)", re.DOTALL)
_INSTRUCTION = re.compile(
    r"^0x(?P<address>[0-9a-fA-F]{8}):\s+(?P<disassembly>.+?)\s*$",
    re.MULTILINE,
)
_LONG = re.compile(r"^\.long\s+0x(?P<word>[0-9a-fA-F]{8})$")


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(name: str, data: bytes) -> dict[str, object]:
    return {"name": name, "size": len(data), "sha256": _sha256(data)}


def _parse_blocks(raw: bytes) -> list[list[dict[str, object]]]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise OpenRiscDiscoveryError("trace is not ASCII") from exc
    blocks: list[list[dict[str, object]]] = []
    for block_match in _BLOCK.finditer(text):
        instructions = [
            {
                "address": int(match.group("address"), 16),
                "disassembly": match.group("disassembly").rstrip(),
            }
            for match in _INSTRUCTION.finditer(block_match.group("body"))
        ]
        if instructions:
            blocks.append(instructions)
    return blocks


def parse_illegal_instruction_boundary(raw: bytes) -> dict[str, object] | None:
    """Return the final raw word immediately before QEMU's illegal vector."""

    blocks = _parse_blocks(raw)
    for index in range(1, len(blocks)):
        if blocks[index][0]["address"] != _EXCEPTION_PC:
            continue
        previous = blocks[index - 1]
        final = previous[-1]
        match = _LONG.fullmatch(str(final["disassembly"]))
        if match is None:
            raise OpenRiscDiscoveryError(
                "exception vector was reached without a final raw instruction"
            )
        address = int(final["address"])
        if address < _RUNTIME_BASE:
            raise OpenRiscDiscoveryError("illegal instruction is below firmware base")
        return {
            "file_offset": address - _RUNTIME_BASE,
            "runtime_address": address,
            "word": int(match.group("word"), 16),
        }
    return None


def build_qemu_discovery_command(
    qemu: str | Path,
    image: str | Path,
    trace: str | Path,
) -> tuple[str, ...]:
    """Build the exact system-emulation command used by discovery."""

    return (
        str(qemu),
        "-M",
        "or1k-sim",
        "-cpu",
        "ambarella-cv22",
        "-m",
        "4G",
        "-nographic",
        "-device",
        f"loader,file={image},addr=0x400000,cpu-num=0,force-raw=on",
        "-d",
        "in_asm",
        "-D",
        str(trace),
    )


def run_qemu_discovery_iteration(
    qemu: str | Path,
    trace_dir: str | Path,
    *,
    timeout_seconds: float,
    iteration: int,
    image: bytes,
) -> DiscoveryTrace:
    """Run one bounded QEMU iteration and preserve every raw output artifact."""

    qemu_path = Path(qemu).resolve()
    destination = Path(trace_dir).resolve()
    if not qemu_path.is_file() or not os.access(qemu_path, os.X_OK):
        raise OpenRiscDiscoveryError(f"QEMU executable is unavailable: {qemu_path}")
    if (
        not isinstance(iteration, int)
        or isinstance(iteration, bool)
        or iteration < 0
    ):
        raise OpenRiscDiscoveryError("iteration must be a non-negative integer")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise OpenRiscDiscoveryError("timeout_seconds must be positive")
    if not isinstance(image, bytes) or not image:
        raise OpenRiscDiscoveryError("probe image must be non-empty bytes")

    destination.mkdir(parents=True, exist_ok=True)
    image_path = destination / f"probe-{iteration:03d}.bin"
    trace_path = destination / f"trace-{iteration:03d}.log"
    stdout_path = destination / f"stdout-{iteration:03d}.bin"
    stderr_path = destination / f"stderr-{iteration:03d}.bin"
    image_path.write_bytes(image)
    command = build_qemu_discovery_command(qemu_path, image_path, trace_path)
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout_seconds),
        )
        stdout = completed.stdout
        stderr = completed.stderr
        outcome = "exited"
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.output or b""
        stderr = exc.stderr or b""
        outcome = "timeout"
        returncode = None
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    try:
        raw_trace = trace_path.read_bytes()
    except OSError as exc:
        raise OpenRiscDiscoveryError(
            f"QEMU did not produce trace {trace_path}: {exc}"
        ) from exc
    return DiscoveryTrace(
        raw=raw_trace,
        runner={
            "outcome": outcome,
            "returncode": returncode,
            "timeout_seconds": float(timeout_seconds),
            "qemu_sha256": _sha256(qemu_path.read_bytes()),
            "command": list(command),
            "artifacts": {
                "probe_image": _artifact(image_path.name, image),
                "trace": _artifact(trace_path.name, raw_trace),
                "stdout": _artifact(stdout_path.name, stdout),
                "stderr": _artifact(stderr_path.name, stderr),
            },
        },
    )


def _derive_image(source: bytes, patches: list[dict[str, int]]) -> bytes:
    derived = bytearray(source)
    for patch in patches:
        offset = patch["file_offset"]
        observed = struct.unpack_from("<I", derived, offset)[0]
        if observed != patch["word"]:
            raise OpenRiscDiscoveryError(
                f"probe source word changed at {_hex32(offset)}"
            )
        struct.pack_into("<I", derived, offset, _NOP)
    return word_swap_32(bytes(derived))


def discover_openrisc_boundaries(
    firmware_path: str | Path,
    provider: TraceProvider,
    *,
    max_iterations: int,
) -> dict[str, Any]:
    """Discover a bounded sequence of illegal-instruction control boundaries."""

    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise OpenRiscDiscoveryError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise OpenRiscDiscoveryError("max_iterations must be positive")
    firmware = analyze_openrisc_firmware(firmware_path)
    try:
        source = Path(firmware_path).resolve().read_bytes()
    except OSError as exc:
        raise OpenRiscDiscoveryError(f"cannot read firmware: {exc}") from exc

    patches: list[dict[str, int]] = []
    iterations: list[dict[str, object]] = []
    stop_reason = "max_iterations_reached"
    for iteration in range(max_iterations):
        image = _derive_image(source, patches)
        trace = provider(iteration, image)
        if not isinstance(trace, DiscoveryTrace):
            raise OpenRiscDiscoveryError("trace provider returned an invalid result")
        boundary = parse_illegal_instruction_boundary(trace.raw)
        record: dict[str, object] = {
            "iteration": iteration,
            "probe_image": {"size": len(image), "sha256": _sha256(image)},
            "trace": {"size": len(trace.raw), "sha256": _sha256(trace.raw)},
            "runner": trace.runner,
        }
        if boundary is None:
            record["instruction"] = None
            iterations.append(record)
            if trace.runner.get("outcome") == "timeout":
                stop_reason = "timeout_without_illegal_instruction_boundary"
            else:
                stop_reason = "no_illegal_instruction_boundary_observed"
            break

        offset = int(boundary["file_offset"])
        if offset < 0 or offset + 4 > len(source) or offset % 4:
            raise OpenRiscDiscoveryError(
                f"boundary offset {_hex32(offset)} is outside aligned firmware"
            )
        source_word = struct.unpack_from("<I", source, offset)[0]
        traced_word = int(boundary["word"])
        if source_word != traced_word:
            raise OpenRiscDiscoveryError(
                f"trace word {_hex32(traced_word)} differs from firmware word "
                f"{_hex32(source_word)} at {_hex32(offset)}"
            )
        if any(patch["file_offset"] == offset for patch in patches):
            raise OpenRiscDiscoveryError(
                f"trace repeated already patched boundary {_hex32(offset)}"
            )
        instruction = {
            "file_offset": _hex32(offset),
            "runtime_address": _hex32(int(boundary["runtime_address"])),
            "word": _hex32(source_word),
            "opcode": f"0x{source_word >> 26:02x}",
        }
        record["instruction"] = instruction
        iterations.append(record)
        patches.append({"file_offset": offset, "word": source_word})

    return {
        "schema": "verkeye.compat.openrisc-discovery.v1",
        "firmware": firmware["artifact"],
        "iterations": iterations,
        "discovered_instruction_count": len(patches),
        "stop_reason": stop_reason,
        "state_transitions_resolved": False,
        "fidelity_gate": "failed_by_construction",
        "claim_scope": (
            "Each iteration replaces only earlier trace-proven trapping words "
            "with l.nop 0 to inventory later control-flow boundaries. The derived "
            "execution is non-faithful and proves no implementation-specific "
            "instruction semantics, "
            "accelerator behavior, tensor value, or inference result."
        ),
    }
