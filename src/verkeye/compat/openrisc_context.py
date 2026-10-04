"""Capture fail-closed CPU context at CV22 OpenRISC trap boundaries.

The first context is exact up to the first unsupported instruction.  Every
later context is explicitly tainted by the earlier ``l.nop 0`` substitutions
used by boundary discovery.  Register snapshots are evidence for subsequent
semantic work; they do not resolve any custom instruction by themselves.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


class OpenRiscContextError(ValueError):
    """A CPU trace or its pinned discovery artifact violates the contract."""


@dataclass(frozen=True, slots=True)
class CpuTrace:
    """Raw QEMU CPU trace and deterministic runner provenance."""

    raw: bytes
    runner: dict[str, object]


ContextProvider = Callable[[int, Path], CpuTrace]

_EXCEPTION_PC = 0x0000_0700
_BLOCK = re.compile(
    r"(?:^|----------------\n)IN:\s*\n(?P<body>.*?)(?=----------------\nIN:|\Z)",
    re.DOTALL,
)
_INSTRUCTION = re.compile(
    r"^0x(?P<address>[0-9a-fA-F]{8}):\s+(?P<disassembly>.+?)\s*$",
    re.MULTILINE,
)
_REGISTER = re.compile(r"\bR(?P<index>\d{2})=(?P<value>[0-9a-fA-F]{8})\b")
_LONG = re.compile(r"^\.long\s+0x(?P<word>[0-9a-fA-F]{8})$")


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def sha256(data: bytes) -> str:
    """Return a lowercase SHA-256 digest for an in-memory artifact."""

    if not isinstance(data, bytes):
        raise OpenRiscContextError("artifact content must be bytes")
    return hashlib.sha256(data).hexdigest()


def artifact(name: str, data: bytes) -> dict[str, object]:
    """Describe an artifact without embedding host-specific paths."""

    if not isinstance(name, str) or not name:
        raise OpenRiscContextError("artifact name must be a non-empty string")
    return {"name": name, "size": len(data), "sha256": sha256(data)}


def _parse_int32(value: object, *, field: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]{1,8}", value):
        raise OpenRiscContextError(f"{field} must be a hexadecimal 32-bit value")
    return int(value, 16)


def _trace_blocks(raw: bytes) -> list[dict[str, object]]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise OpenRiscContextError("CPU trace is not ASCII") from exc
    blocks: list[dict[str, object]] = []
    for match in _BLOCK.finditer(text):
        body = match.group("body")
        instructions = [
            {
                "runtime_address": int(item.group("address"), 16),
                "disassembly": item.group("disassembly").rstrip(),
            }
            for item in _INSTRUCTION.finditer(body)
        ]
        if not instructions:
            continue
        registers = {
            int(item.group("index")): int(item.group("value"), 16)
            for item in _REGISTER.finditer(body)
        }
        blocks.append({"instructions": instructions, "registers": registers})
    return blocks


def parse_cpu_boundary_trace(
    raw: bytes,
    *,
    expected_address: int,
    expected_word: int,
) -> dict[str, object]:
    """Parse the exact boundary block and exception-entry register snapshot."""

    if not isinstance(raw, bytes) or not raw:
        raise OpenRiscContextError("CPU trace must be non-empty bytes")
    for label, value in (
        ("expected_address", expected_address),
        ("expected_word", expected_word),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFF_FFFF:
            raise OpenRiscContextError(f"{label} must be a 32-bit integer")

    blocks = _trace_blocks(raw)
    for index in range(1, len(blocks)):
        exception = blocks[index]
        exception_instructions = exception["instructions"]
        if exception_instructions[0]["runtime_address"] != _EXCEPTION_PC:
            continue
        boundary = blocks[index - 1]
        boundary_instructions = boundary["instructions"]
        final = boundary_instructions[-1]
        if final["runtime_address"] != expected_address:
            continue
        raw_word = _LONG.fullmatch(str(final["disassembly"]))
        if raw_word is None or int(raw_word.group("word"), 16) != expected_word:
            raise OpenRiscContextError("boundary raw word differs from discovery record")
        registers = exception["registers"]
        if set(registers) != set(range(32)):
            raise OpenRiscContextError(
                "exception entry lacks a complete register state (r00-r31)"
            )
        return {
            "translation_block": [
                {
                    "runtime_address": _hex32(int(item["runtime_address"])),
                    "disassembly": str(item["disassembly"]),
                }
                for item in boundary_instructions
            ],
            "boundary_pc": _hex32(expected_address),
            "next_pc": _hex32(_EXCEPTION_PC),
            "qemu_result": "illegal_instruction_exception",
            "register_state_before_boundary": {
                f"r{register:02d}": _hex32(registers[register])
                for register in range(32)
            },
        }
    raise OpenRiscContextError("expected illegal-instruction boundary was not observed")


def build_qemu_cpu_trace_command(
    qemu: str | Path,
    image: str | Path,
    trace: str | Path,
) -> tuple[str, ...]:
    """Build the pinned QEMU command for instruction and CPU-state logging."""

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
        "in_asm,cpu",
        "-D",
        str(trace),
    )


def run_qemu_cpu_trace_iteration(
    qemu: str | Path,
    trace_dir: str | Path,
    *,
    timeout_seconds: float,
    iteration: int,
    probe: str | Path,
) -> CpuTrace:
    """Run one bounded context capture and preserve all raw outputs."""

    qemu_path = Path(qemu).resolve()
    probe_path = Path(probe).resolve()
    destination = Path(trace_dir).resolve()
    if not qemu_path.is_file() or not os.access(qemu_path, os.X_OK):
        raise OpenRiscContextError(f"QEMU executable is unavailable: {qemu_path}")
    if not probe_path.is_file():
        raise OpenRiscContextError(f"probe image is unavailable: {probe_path}")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise OpenRiscContextError("iteration must be a non-negative integer")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise OpenRiscContextError("timeout_seconds must be positive")

    destination.mkdir(parents=True, exist_ok=True)
    trace_path = destination / f"cpu-trace-{iteration:03d}.log"
    stdout_path = destination / f"stdout-{iteration:03d}.bin"
    stderr_path = destination / f"stderr-{iteration:03d}.bin"
    command = build_qemu_cpu_trace_command(qemu_path, probe_path, trace_path)
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
        raise OpenRiscContextError(
            f"QEMU did not produce CPU trace {trace_path}: {exc}"
        ) from exc
    return CpuTrace(
        raw=raw_trace,
        runner={
            "outcome": outcome,
            "returncode": returncode,
            "timeout_seconds": float(timeout_seconds),
            "qemu_sha256": sha256(qemu_path.read_bytes()),
            "command": list(command),
            "artifacts": {
                "trace": artifact(trace_path.name, raw_trace),
                "stdout": artifact(stdout_path.name, stdout),
                "stderr": artifact(stderr_path.name, stderr),
            },
        },
    )


def _validate_probe(
    probe_dir: Path,
    record: Mapping[str, object],
) -> Path:
    runner = record.get("runner")
    if not isinstance(runner, Mapping):
        raise OpenRiscContextError("discovery record lacks runner provenance")
    artifacts = runner.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise OpenRiscContextError("discovery record lacks probe artifacts")
    expected = artifacts.get("probe_image")
    if not isinstance(expected, Mapping):
        raise OpenRiscContextError("discovery record lacks probe artifact")
    name = expected.get("name")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise OpenRiscContextError("probe artifact name is invalid")
    path = probe_dir / name
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise OpenRiscContextError(f"probe artifact is unavailable: {name}") from exc
    observed = artifact(name, content)
    if (
        expected.get("size") != observed["size"]
        or expected.get("sha256") != observed["sha256"]
    ):
        raise OpenRiscContextError(f"probe artifact integrity failed: {name}")
    return path


def capture_openrisc_contexts(
    discovery: Mapping[str, Any],
    probe_dir: str | Path,
    provider: ContextProvider,
) -> dict[str, object]:
    """Capture the complete context for each trace-proven boundary."""

    if discovery.get("fidelity_gate") != "failed_by_construction":
        raise OpenRiscContextError("discovery must retain failed_by_construction")
    iterations = discovery.get("iterations")
    if not isinstance(iterations, list):
        raise OpenRiscContextError("discovery iterations must be a list")
    base = Path(probe_dir).resolve()
    contexts: list[dict[str, object]] = []
    for expected_iteration, record in enumerate(iterations):
        if not isinstance(record, Mapping):
            raise OpenRiscContextError("discovery iteration must be an object")
        instruction = record.get("instruction")
        if instruction is None:
            continue
        if not isinstance(instruction, Mapping):
            raise OpenRiscContextError("discovery instruction must be an object")
        iteration = record.get("iteration")
        if iteration != expected_iteration:
            raise OpenRiscContextError("discovery iterations are not contiguous")
        probe = _validate_probe(base, record)
        trace = provider(expected_iteration, probe)
        if not isinstance(trace, CpuTrace):
            raise OpenRiscContextError("context provider returned an invalid result")
        address = _parse_int32(instruction.get("runtime_address"), field="runtime_address")
        word = _parse_int32(instruction.get("word"), field="word")
        parsed = parse_cpu_boundary_trace(
            trace.raw,
            expected_address=address,
            expected_word=word,
        )
        fidelity = (
            "exact_until_first_unsupported_instruction"
            if expected_iteration == 0
            else f"altered_by_{expected_iteration}_prior_nop_substitution"
        )
        contexts.append(
            {
                "iteration": expected_iteration,
                "instruction": dict(instruction),
                "probe_image": artifact(probe.name, probe.read_bytes()),
                "cpu_trace": artifact(
                    str(
                        trace.runner.get("artifacts", {})
                        .get("trace", {})
                        .get("name", f"cpu-trace-{expected_iteration:03d}.log")
                    ),
                    trace.raw,
                ),
                "prior_nop_substitution_count": expected_iteration,
                "pre_boundary_state_fidelity": fidelity,
                **parsed,
                "runner": trace.runner,
            }
        )

    return {
        "schema": "verkeye.compat.openrisc-context.v1",
        "context_count": len(contexts),
        "contexts": contexts,
        "state_transitions_resolved": False,
        "fidelity_gate": "failed_by_construction",
        "claim_scope": (
            "The exception-entry GPR snapshot records pre-boundary register state. "
            "Only iteration 0 is exact up to its unsupported instruction; later "
            "snapshots are altered by explicitly counted prior l.nop substitutions. "
            "No custom instruction semantics or tensor values are inferred."
        ),
    }
