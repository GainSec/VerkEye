"""Build and parse the recovered NNCtrl userspace ABI probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


class ProbeError(ValueError):
    """A native probe build or record stream is not trustworthy."""


PROBE_PREFIX = "VERKEYE_PROBE_JSON "


def build_probe_compile_command(
    compiler: str | Path,
    source: str | Path,
    output: str | Path,
) -> tuple[str, ...]:
    compiler_path = Path(compiler).resolve()
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not compiler_path.is_file():
        raise ProbeError(f"cross compiler is unavailable: {compiler_path}")
    if not source_path.is_file():
        raise ProbeError(f"native probe source is unavailable: {source_path}")
    return (
        str(compiler_path),
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fstack-protector-strong",
        "-D_FORTIFY_SOURCE=2",
        str(source_path),
        "-o",
        str(output_path),
        "-ldl",
    )


def build_shim_compile_command(
    compiler: str | Path,
    source: str | Path,
    output: str | Path,
) -> tuple[str, ...]:
    compiler_path = Path(compiler).resolve()
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not compiler_path.is_file():
        raise ProbeError(f"cross compiler is unavailable: {compiler_path}")
    if not source_path.is_file():
        raise ProbeError(f"Cavalry shim source is unavailable: {source_path}")
    return (
        str(compiler_path),
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fstack-protector-strong",
        "-D_FORTIFY_SOURCE=2",
        "-fPIC",
        "-shared",
        str(source_path),
        "-o",
        str(output_path),
        "-ldl",
    )


def parse_probe_records(stdout: str) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith(PROBE_PREFIX):
            continue
        try:
            record = json.loads(line[len(PROBE_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise ProbeError(f"invalid probe JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ProbeError("probe record must be an object")
        if record.get("schema") != "verkeye.cv22.native-probe.v1":
            raise ProbeError("unsupported probe schema")
        if record.get("sequence") != len(records):
            raise ProbeError("non-contiguous probe sequence")
        if not isinstance(record.get("operation"), str):
            raise ProbeError("probe operation must be a string")
        if record.get("status") not in {"passed", "failed"}:
            raise ProbeError("probe status must be passed or failed")
        records.append(record)
    if not records:
        raise ProbeError("no structured probe records found")
    return tuple(records)


def _fnv1a64(data: bytes) -> str:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFF_FFFF_FFFF_FFFF
    return f"{value:016x}"


def verify_probe_memory_dump(
    records: Sequence[dict[str, Any]], dump_path: str | Path
) -> dict[str, object]:
    successful_dumps = [
        record
        for record in records
        if record.get("operation") == "nnctrl_memory_dump"
        and record.get("status") == "passed"
        and record.get("return_code") == 0
    ]
    if len(successful_dumps) != 1:
        raise ProbeError("expected exactly one successful memory dump record")
    successful_memories = [
        record
        for record in records
        if record.get("operation") == "nnctrl_memory"
        and record.get("status") == "passed"
        and record.get("return_code") == 0
    ]
    if len(successful_memories) != 1:
        raise ProbeError("expected exactly one successful loaded-memory record")

    destination = Path(dump_path).resolve()
    try:
        data = destination.read_bytes()
    except OSError as exc:
        raise ProbeError(f"cannot read working-memory dump: {exc}") from exc
    dump_record = successful_dumps[0]
    memory_record = successful_memories[0]
    expected_sizes = {dump_record.get("size"), memory_record.get("size")}
    if len(expected_sizes) != 1 or len(data) not in expected_sizes:
        raise ProbeError(
            "working-memory size mismatch: "
            f"records={sorted(str(value) for value in expected_sizes)}, "
            f"observed={len(data)}"
        )
    digest = _fnv1a64(data)
    expected_digests = {
        dump_record.get("fnv1a64"),
        memory_record.get("fnv1a64"),
    }
    if len(expected_digests) != 1 or digest not in expected_digests:
        raise ProbeError(
            "working-memory FNV-1a mismatch: "
            f"records={sorted(str(value) for value in expected_digests)}, "
            f"observed={digest}"
        )
    return {
        "schema": "verkeye.cv22.working-memory-evidence.v1",
        "path": str(destination),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "fnv1a64": digest,
        "status": "verified",
    }
