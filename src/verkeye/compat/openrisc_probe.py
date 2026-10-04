"""Build explicitly non-faithful OpenRISC control-flow probe images."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .ambarella_custom import analyze_ambarella_custom
from .openrisc import word_swap_32


class OpenRiscProbeError(ValueError):
    """A requested probe would exceed the pinned evidence boundary."""


_NOP = 0x1500_0000


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class OpenRiscProbe:
    """A derived image and provenance manifest for a control-flow experiment."""

    little_endian_image: bytes
    qemu_word_swapped_image: bytes
    report: dict[str, Any]


def build_openrisc_probe(
    firmware_path: str | Path, patch_offsets: Iterable[int]
) -> OpenRiscProbe:
    """Replace selected inventoried vendor words with ``l.nop 0``.

    This operation is only a reachability experiment.  The report says so
    explicitly, and no result from this derived image may satisfy a fidelity
    or inference-execution gate.
    """

    offsets = tuple(patch_offsets)
    if not offsets:
        raise OpenRiscProbeError("probe requires at least one patch offset")
    if len(offsets) != len(set(offsets)):
        raise OpenRiscProbeError("duplicate patch offset")
    if any(
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or offset % 4
        for offset in offsets
    ):
        raise OpenRiscProbeError("patch offsets must be aligned non-negative integers")

    source = Path(firmware_path).resolve()
    inventory = analyze_ambarella_custom(source)
    by_offset = {
        int(item["file_offset"], 16): item for item in inventory["instructions"]
    }
    by_offset.update(
        {
            int(item["file_offset"], 16): item
            for item in inventory["dynamically_reached_instructions"]
        }
    )
    for offset in offsets:
        if offset not in by_offset:
            raise OpenRiscProbeError(
                f"offset {_hex32(offset)} is not an inventoried unsupported instruction"
            )

    try:
        source_data = source.read_bytes()
    except OSError as exc:
        raise OpenRiscProbeError(f"cannot read firmware {source}: {exc}") from exc
    derived = bytearray(source_data)
    patches: list[dict[str, str]] = []
    for offset in sorted(offsets):
        original_word = struct.unpack_from("<I", derived, offset)[0]
        expected_word = int(by_offset[offset]["word"], 16)
        if original_word != expected_word:
            raise OpenRiscProbeError(
                f"word at {_hex32(offset)} changed before probe construction"
            )
        struct.pack_into("<I", derived, offset, _NOP)
        patches.append(
            {
                "file_offset": _hex32(offset),
                "runtime_address": _hex32(0x0040_0000 + offset),
                "original_word": _hex32(original_word),
                "replacement_word": _hex32(_NOP),
                "replacement_mnemonic": "l.nop 0",
            }
        )

    little_endian = bytes(derived)
    qemu_image = word_swap_32(little_endian)
    report = {
        "schema": "verkeye.compat.openrisc-probe.v1",
        "source_artifact": inventory["artifact"],
        "patches": patches,
        "derived_little_endian_image": {
            "size": len(little_endian),
            "sha256": _sha256(little_endian),
        },
        "qemu_word_swapped_image": {
            "size": len(qemu_image),
            "sha256": _sha256(qemu_image),
        },
        "interpretation": "control_flow_probe_not_faithful_execution",
        "fidelity_gate": "failed_by_construction",
        "claim_scope": (
            "Selected unresolved implementation-specific instructions are replaced "
            "with l.nop 0 "
            "only to reveal later control-flow boundaries. The derived image does "
            "not preserve implementation-specific state transitions and cannot "
            "prove inference."
        ),
    }
    return OpenRiscProbe(
        little_endian_image=little_endian,
        qemu_word_swapped_image=qemu_image,
        report=report,
    )
