"""Exact loader and image-format evidence for the recovered Cavalry firmware."""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CavalryFirmwareError(ValueError):
    """The firmware or loader contradicts the pinned recovered artifacts."""


_FIRMWARE_SIZE = 101_332
_FIRMWARE_SHA256 = "889c32b8b599454b1a98dfac4e6f9ee7f25abe8f02e21030646e697b07aa45f3"
_LOADER_SIZE = 67_736
_LOADER_SHA256 = "d19a8d03f7cbc8e371bef1f5c653de400f117cd82d15a6b4e8b9f92d7f0c1ad6"
_VERSION_OFFSET = 0x40
_VERSION_SIZE = 20
_CRC_SIZE = 4
_PLATFORMS = {0: "CV22", 1: "CV25"}
_STRING_ANCHORS = (
    (0xFC14, "CMD 0x%08x is not supported."),
    (0xFC32, "STOP_CMD is found."),
    (0xFC46, "DAG_RUN_CMD is found."),
    (0xFC5D, "CREATE_SESSION_CMD is found."),
    (0xFC7B, "SET_LOG_LEVEL_CMD is found."),
    (0xFCBE, "HOTLINK_RUN_CMD is found."),
    (0xFDA6, "Cavalry ucode is started."),
    (0xFDC1, "CMD Q Addr 0x%x, size 0x%x."),
    (0xFDDE, "MSG Q Addr 0x%x, size 0x%x."),
    (0xFDFB, "LOG Q Addr 0x%x, size 0x%x."),
    (0x109C5, "Cavalry Scheduler Exiting..."),
    (0x109E3, "Cavalry Scheduler Started."),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class CavalryFirmwareVersion:
    platform_id: int
    platform_name: str
    version: int
    build_date: int
    source_revision: int
    hotlink_firmware_version: int

    @property
    def rendered(self) -> str:
        return (
            f"{self.platform_name}-Ver.{self.version}-{self.build_date:08d}-"
            f"{self.source_revision:08x}-hl_fw.{self.hotlink_firmware_version:08d}"
        )

    def to_document(self) -> dict[str, object]:
        return {
            "offset": _VERSION_OFFSET,
            "platform_id": self.platform_id,
            "platform_name": self.platform_name,
            "version": self.version,
            "build_date": self.build_date,
            "source_revision": f"0x{self.source_revision:08x}",
            "hotlink_firmware_version": self.hotlink_firmware_version,
            "rendered": self.rendered,
        }


@dataclass(frozen=True, slots=True)
class CavalryFirmwareImage:
    data: bytes
    sha256: str
    stored_crc32: int
    computed_crc32: int
    version: CavalryFirmwareVersion

    @property
    def payload_size(self) -> int:
        return len(self.data) - _CRC_SIZE


def parse_cavalry_firmware(data: bytes) -> CavalryFirmwareImage:
    """Parse only the header and checksum semantics evidenced by cavalry_load."""

    minimum = _VERSION_OFFSET + _VERSION_SIZE + _CRC_SIZE
    if len(data) < minimum:
        raise CavalryFirmwareError(
            f"Cavalry firmware is too short: expected at least {minimum} bytes, "
            f"observed {len(data)}"
        )

    platform_id, version, build_date, revision, hotlink = struct.unpack_from(
        "<5I", data, _VERSION_OFFSET
    )
    if (
        platform_id not in _PLATFORMS
        or version == 0
        or not 20_000_101 <= build_date <= 21_001_231
    ):
        raise CavalryFirmwareError(
            "Cavalry version header is not an evidenced CV22/CV25 record"
        )

    stored_crc32 = struct.unpack_from("<I", data, len(data) - _CRC_SIZE)[0]
    computed_crc32 = zlib.crc32(data[:-_CRC_SIZE]) & 0xFFFF_FFFF
    if stored_crc32 != computed_crc32:
        raise CavalryFirmwareError(
            "Cavalry firmware CRC32 mismatch: "
            f"stored 0x{stored_crc32:08x}, computed 0x{computed_crc32:08x}"
        )

    return CavalryFirmwareImage(
        data=data,
        sha256=_sha256(data),
        stored_crc32=stored_crc32,
        computed_crc32=computed_crc32,
        version=CavalryFirmwareVersion(
            platform_id=platform_id,
            platform_name=_PLATFORMS[platform_id],
            version=version,
            build_date=build_date,
            source_revision=revision,
            hotlink_firmware_version=hotlink,
        ),
    )


def _read_exact_artifact(
    path: str | Path,
    *,
    role: str,
    expected_size: int,
    expected_sha256: str,
) -> bytes:
    source = Path(path).resolve()
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise CavalryFirmwareError(f"cannot read {role} {source}: {exc}") from exc
    if len(data) != expected_size:
        raise CavalryFirmwareError(
            f"{role} size mismatch: expected {expected_size}, observed {len(data)}"
        )
    digest = _sha256(data)
    if digest != expected_sha256:
        raise CavalryFirmwareError(
            f"{role} sha256 mismatch: expected {expected_sha256}, observed {digest}"
        )
    return data


def analyze_cavalry_firmware(
    firmware_path: str | Path,
    loader_path: str | Path,
) -> dict[str, Any]:
    """Return the exact camera firmware/loader contract with bounded claims."""

    firmware_data = _read_exact_artifact(
        firmware_path,
        role="cavalry firmware",
        expected_size=_FIRMWARE_SIZE,
        expected_sha256=_FIRMWARE_SHA256,
    )
    loader_data = _read_exact_artifact(
        loader_path,
        role="cavalry loader",
        expected_size=_LOADER_SIZE,
        expected_sha256=_LOADER_SHA256,
    )
    image = parse_cavalry_firmware(firmware_data)

    anchors: list[dict[str, object]] = []
    for expected_offset, text in _STRING_ANCHORS:
        encoded = text.encode("ascii")
        observed_offset = firmware_data.find(encoded)
        if observed_offset != expected_offset:
            raise CavalryFirmwareError(
                f"firmware string anchor {text!r} expected at 0x{expected_offset:x}, "
                f"observed at {observed_offset}"
            )
        anchors.append({"offset": expected_offset, "text": text})

    return {
        "schema": "verkeye.compat.cavalry-firmware.v1",
        "firmware": {
            "path_role": "cavalry_firmware",
            "size": len(firmware_data),
            "sha256": image.sha256,
            "payload_size": image.payload_size,
            "stored_crc32": f"0x{image.stored_crc32:08x}",
            "computed_crc32": f"0x{image.computed_crc32:08x}",
            "crc_byte_order": "little",
        },
        "loader": {
            "path_role": "cavalry_load",
            "size": len(loader_data),
            "sha256": _sha256(loader_data),
        },
        "version": image.version.to_document(),
        "load_contract": {
            "query_ioctl": "0xc0084300",
            "query_buffer_index": 8,
            "query_structure_size": 24,
            "query_size_field_offset": 8,
            "query_physical_offset_field_offset": 16,
            "mmap_protection": "PROT_WRITE",
            "mmap_flags": "MAP_SHARED",
            "file_copy_offset": 0,
            "crc_span": {"start": 0, "end_exclusive": image.payload_size},
            "crc_field_offset": image.payload_size,
            "version_header_offset": _VERSION_OFFSET,
            "stop_ioctl": "0xc0084302",
            "start_ioctl": "0xc0084301",
            "provenance": {
                "map_function": "cavalry_load VA 0x1aac",
                "load_function": "cavalry_load VA 0x1de0",
                "version_function": "cavalry_load VA 0x19d0",
                "start_callsite": "cavalry_load VA 0x28fc",
                "driver_query_function": (
                    "cavalry.ko cavalry_query_buf VA 0x0d50"
                ),
            },
        },
        "string_anchors": anchors,
        "classification": {
            "role": "CV22 Cavalry neural scheduler firmware",
            "instruction_set": (
                "OpenRISC 1000 little-endian with Ambarella extensions"
            ),
            "inference_execution_supported": False,
            "unresolved_boundary": (
                "Ambarella OpenRISC extension semantics and accelerator state "
                "transitions"
            ),
        },
    }
