"""Deterministic static observations from recovered CV22 ELF binaries."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence


class VendorRuntimeError(ValueError):
    """A vendor artifact or analysis tool could not be trusted."""


def supports_static_elf_analysis(architecture: str) -> bool:
    """Return whether a corpus architecture denotes a Linux AArch64 ELF."""

    return architecture == "linux-aarch64" or architecture.startswith(
        "linux-aarch64-"
    )


_HEADER_FIELD = re.compile(r"^\s*([^:]+):\s*(.*?)\s*$")
_SECTION = re.compile(
    r"^\s*\[\s*(?P<index>\d+)\]\s+"
    r"(?P<name>\S+)\s+(?P<type>\S+)\s+"
    r"(?P<address>[0-9a-fA-F]+)\s+(?P<offset>[0-9a-fA-F]+)\s+"
    r"(?P<size>[0-9a-fA-F]+)\s+(?P<entry>[0-9a-fA-F]+)\s+"
    r"(?P<flags>\S*)"
)
_NEEDED = re.compile(r"\(NEEDED\)\s+Shared library: \[(?P<name>[^]]+)\]")
_SYMBOL = re.compile(
    r"^\s*(?P<index>\d+):\s+"
    r"(?P<value>[0-9a-fA-F]+)\s+(?P<size>\d+)\s+"
    r"(?P<type>\S+)\s+(?P<bind>\S+)\s+(?P<visibility>\S+)\s+"
    r"(?P<section>\S+)\s*(?P<name>.*)$"
)
_SYMBOL_TABLE = re.compile(r"^Symbol table '(?P<name>[^']+)' contains \d+ entries:")
_API_STRING = re.compile(
    r"(?:nnctrl|cavalry|CAVALRY_|/dev/|vproc|dewarp)", re.IGNORECASE
)
_NAMED_IOCTL = re.compile(r"^CAVALRY_[A-Z0-9_]+$")


def run_tool(argv: Sequence[str]) -> str:
    """Run one analysis command and reject non-zero or undecodable output."""

    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, UnicodeError) as exc:
        raise VendorRuntimeError(f"cannot run {' '.join(argv)}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise VendorRuntimeError(
            f"{' '.join(argv)} failed with exit status {completed.returncode}: {detail}"
        )
    return completed.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _printable_strings(data: bytes, minimum: int = 4) -> tuple[dict[str, Any], ...]:
    observations: list[dict[str, Any]] = []
    start: int | None = None
    for index in range(len(data) + 1):
        value = data[index] if index < len(data) else 0
        if 0x20 <= value <= 0x7E:
            if start is None:
                start = index
            continue
        if start is not None and index - start >= minimum:
            raw = data[start:index]
            text = raw.decode("ascii")
            if _API_STRING.search(text):
                observations.append(
                    {
                        "file_offset": start,
                        "size": len(raw),
                        "value": text,
                        "provenance": "contiguous printable bytes in ELF file",
                    }
                )
        start = None
    return tuple(observations)


def _header(output: str) -> dict[str, str]:
    wanted = {"Class", "Data", "OS/ABI", "Type", "Machine"}
    result: dict[str, str] = {}
    for line in output.splitlines():
        match = _HEADER_FIELD.match(line)
        if match and match.group(1).strip() in wanted:
            result[match.group(1).strip()] = match.group(2).strip()
    missing = wanted - result.keys()
    if missing:
        raise VendorRuntimeError(
            f"readelf output lacks header fields: {', '.join(sorted(missing))}"
        )
    return {
        "class": result["Class"],
        "data": result["Data"],
        "os_abi": result["OS/ABI"],
        "type": result["Type"],
        "machine": result["Machine"],
    }


def _sections(output: str) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for line in output.splitlines():
        match = _SECTION.match(line)
        if not match:
            continue
        flags = match.group("flags")
        classification = "code" if "X" in flags else "data" if "A" in flags else "metadata"
        result.append(
            {
                "index": int(match.group("index")),
                "name": match.group("name"),
                "type": match.group("type"),
                "virtual_address": int(match.group("address"), 16),
                "file_offset": int(match.group("offset"), 16),
                "size": int(match.group("size"), 16),
                "entry_size": int(match.group("entry"), 16),
                "flags": flags,
                "classification": classification,
                "provenance": "ELF section header",
            }
        )
    if not result:
        raise VendorRuntimeError("readelf output contains no section headers")
    return tuple(result)


def _clean_symbol_name(raw: str) -> str:
    name = raw.strip().split(" ", 1)[0]
    return name.split("@", 1)[0]


def _symbols(
    output: str,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    imported: list[dict[str, Any]] = []
    exported: list[dict[str, Any]] = []
    defined: list[dict[str, Any]] = []
    dynamic_seen: set[tuple[str, str, int, int]] = set()
    defined_seen: set[tuple[str, str, int, int]] = set()
    symbol_table: str | None = None
    for line in output.splitlines():
        table_match = _SYMBOL_TABLE.match(line)
        if table_match:
            symbol_table = table_match.group("name")
            continue
        match = _SYMBOL.match(line)
        if not match or symbol_table is None:
            continue
        name = _clean_symbol_name(match.group("name"))
        if not name:
            continue
        section = match.group("section")
        address = int(match.group("value"), 16)
        size = int(match.group("size"))
        record = {
            "name": name,
            "virtual_address": address,
            "size": size,
            "type": match.group("type"),
            "binding": match.group("bind"),
            "visibility": match.group("visibility"),
            "section": section,
            "provenance": f"ELF {symbol_table} virtual address",
        }
        key = (section, name, address, size)
        if section != "UND" and key not in defined_seen:
            defined.append(record)
            defined_seen.add(key)
        if symbol_table == ".dynsym" and key not in dynamic_seen:
            if section == "UND":
                imported.append(record)
            elif match.group("bind") in {"GLOBAL", "WEAK"}:
                exported.append(record)
            dynamic_seen.add(key)
    return (
        tuple(sorted(imported, key=lambda item: (item["name"], item["virtual_address"]))),
        tuple(sorted(exported, key=lambda item: (item["name"], item["virtual_address"]))),
        tuple(
            sorted(
                defined,
                key=lambda item: (
                    item["virtual_address"],
                    item["name"],
                    item["section"],
                ),
            )
        ),
    )


def analyze_elf(path: str | Path) -> dict[str, Any]:
    """Return normalized, provenance-labeled observations for an ELF file."""

    source = Path(path).resolve()
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise VendorRuntimeError(f"cannot read {source}: {exc}") from exc
    if not data.startswith(b"\x7fELF"):
        raise VendorRuntimeError(f"{source} is not an ELF file")
    readelf = shutil.which("readelf")
    if readelf is None:
        raise VendorRuntimeError("readelf is required for vendor runtime analysis")
    output = run_tool(
        [
            readelf,
            "--wide",
            "--file-header",
            "--sections",
            "--dynamic",
            "--syms",
            str(source),
        ]
    )
    imported, exported, defined = _symbols(output)
    api_strings = _printable_strings(data)
    named_operations = sorted(
        {
            item["value"]
            for item in api_strings
            if _NAMED_IOCTL.fullmatch(item["value"])
        }
    )
    return {
        "schema": "verkeye.cv22.vendor-runtime.v1",
        "artifact": {
            "path": str(source),
            "size": len(data),
            "sha256": _sha256(data),
        },
        "elf": _header(output),
        "dependencies": sorted(
            {match.group("name") for match in _NEEDED.finditer(output)}
        ),
        "sections": list(_sections(output)),
        "imported_symbols": list(imported),
        "exported_symbols": list(exported),
        "defined_symbols": list(defined),
        "api_strings": list(api_strings),
        "ioctl_evidence": {
            "imported": "ioctl" in {item["name"] for item in imported},
            "named_operations": named_operations,
            "numeric_requests": [],
            "disposition": (
                "named operations are evidenced by file offsets; numeric ioctl "
                "request values remain unresolved"
            ),
        },
    }
