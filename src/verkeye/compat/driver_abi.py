"""Exact ioctl dispatch evidence from the recovered CB62 Cavalry driver."""

from __future__ import annotations

import hashlib
import re
import shutil
import struct
from pathlib import Path
from typing import Any

from verkeye.cv22.vendor_runtime import analyze_elf, run_tool


class DriverAbiError(ValueError):
    """The driver artifact or its evidenced dispatch structure is unexpected."""


_DRIVER_SIZE = 154_192
_DRIVER_SHA256 = "39811bf17113e78c7babe9c0a6b8bad53ca80648847c364da5fccb5dd5f0de24"
_DISPATCHER_ADDRESS = 0x300
_DISPATCHER_SIZE = 1040
_BRANCH_ORIGIN = 0x398
_TABLE_ENTRY_COUNT = 0x83
_TABLE_RELOCATION_OFFSET = 0x718
_RELOCATION_SECTION = re.compile(r"^Relocation section '(?P<name>[^']+)'")
_CALL26 = re.compile(
    r"^\s*(?P<offset>[0-9a-fA-F]+)\s+\S+\s+R_AARCH64_CALL26\s+"
    r"\S+\s+(?P<symbol>\S+)\s+\+"
)
_TABLE_RELOCATION = re.compile(
    rf"^0*{_TABLE_RELOCATION_OFFSET:x}\s+\S+\s+R_AARCH64_ABS64\s+"
    r"\S+\s+\.rodata\s+\+\s+0$"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text_relocations(path: Path) -> tuple[dict[int, str], bool]:
    readelf = shutil.which("readelf")
    if readelf is None:
        raise DriverAbiError("readelf is required for driver ABI analysis")
    output = run_tool([readelf, "--wide", "--relocs", str(path)])
    section: str | None = None
    calls: dict[int, str] = {}
    table_relocation = False
    for line in output.splitlines():
        section_match = _RELOCATION_SECTION.match(line)
        if section_match:
            section = section_match.group("name")
            continue
        if section != ".rela.text":
            continue
        if _TABLE_RELOCATION.fullmatch(line.strip()):
            table_relocation = True
        call_match = _CALL26.match(line)
        if call_match:
            calls[int(call_match.group("offset"), 16)] = call_match.group("symbol")
    return calls, table_relocation


def analyze_cavalry_driver(path: str | Path) -> dict[str, Any]:
    """Recover the exact ioctl-to-handler map for the pinned CB62 driver build."""

    source = Path(path).resolve()
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise DriverAbiError(f"cannot read {source}: {exc}") from exc
    digest = _sha256(data)
    if len(data) != _DRIVER_SIZE:
        raise DriverAbiError(
            f"driver size mismatch: expected {_DRIVER_SIZE}, observed {len(data)}"
        )
    if digest != _DRIVER_SHA256:
        raise DriverAbiError(
            f"driver sha256 mismatch: expected {_DRIVER_SHA256}, observed {digest}"
        )

    elf = analyze_elf(source)
    symbols = {item["name"]: item for item in elf["defined_symbols"]}
    dispatcher = symbols.get("cavalry_ioctl")
    if dispatcher is None:
        raise DriverAbiError("driver lacks cavalry_ioctl symbol")
    if (
        dispatcher["virtual_address"] != _DISPATCHER_ADDRESS
        or dispatcher["size"] != _DISPATCHER_SIZE
    ):
        raise DriverAbiError("cavalry_ioctl symbol does not match pinned dispatcher")

    rodata = next(
        (item for item in elf["sections"] if item["name"] == ".rodata"), None
    )
    if rodata is None:
        raise DriverAbiError("driver lacks .rodata jump-table section")
    table_offset = rodata["file_offset"]
    table_size = _TABLE_ENTRY_COUNT * 2
    if table_offset + table_size > len(data):
        raise DriverAbiError("ioctl jump table exceeds driver file")

    calls, table_relocation = _text_relocations(source)
    if not table_relocation:
        raise DriverAbiError("dispatcher does not relocate its table from .rodata+0")

    requests: list[dict[str, Any]] = []
    for index in range(_TABLE_ENTRY_COUNT):
        entry_file_offset = table_offset + index * 2
        relative_words = struct.unpack_from("<h", data, entry_file_offset)[0]
        target = _BRANCH_ORIGIN + relative_words * 4
        callsite = target + 8
        handler = calls.get(callsite)
        if handler is None:
            continue
        requests.append(
            {
                "request": 0xC0084300 + index,
                "handler": handler,
                "dispatch": "jump_table",
                "table_index": index,
                "table_entry_file_offset": entry_file_offset,
                "signed_relative_words": relative_words,
                "dispatch_target": target,
                "handler_callsite": callsite,
                "provenance": (
                    "signed .rodata jump-table entry and .rela.text CALL26 relocation"
                ),
            }
        )

    for request, callsite, expected_handler in (
        (0x80084380, 0x3E0, "cavalry_get_audio_clk"),
        (0x40084381, 0x704, "cavalry_set_cavalry_clk"),
    ):
        handler = calls.get(callsite)
        if handler != expected_handler:
            raise DriverAbiError(
                f"direct ioctl 0x{request:08x} expected {expected_handler}, "
                f"observed {handler!r}"
            )
        requests.append(
            {
                "request": request,
                "handler": handler,
                "dispatch": "direct_compare",
                "handler_callsite": callsite,
                "provenance": (
                    "immediate request comparison and .rela.text CALL26 relocation"
                ),
            }
        )

    return {
        "schema": "verkeye.compat.cavalry-driver-abi.v1",
        "artifact": {"size": len(data), "sha256": digest},
        "dispatcher": {
            "symbol": "cavalry_ioctl",
            "virtual_address": dispatcher["virtual_address"],
            "size": dispatcher["size"],
            "jump_table_section": ".rodata",
            "jump_table_file_offset": table_offset,
            "jump_table_entry_count": _TABLE_ENTRY_COUNT,
            "branch_origin": _BRANCH_ORIGIN,
        },
        "requests": sorted(requests, key=lambda item: item["request"]),
    }
