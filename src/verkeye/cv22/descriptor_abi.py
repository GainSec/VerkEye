"""Source-and-binary proof for the serialized NNCtrl tensor descriptor ABI.

The public source names the fields but omits the proprietary ``cavalry_gen.h``
layout.  The recovered AArch64 library supplies the missing offsets: the
``gen_net_parent_port_size`` implementation loads each field from a fixed
offset in the 0x480-byte serialized descriptor.  This module pins both sides
and fails if either artifact changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from verkeye.compat.vendor_source import verify_source_snapshot

from .vendor_runtime import analyze_elf


class DescriptorABIError(ValueError):
    """The recovered source/binary descriptor proof no longer matches."""


_LIBNNCTRL_SIZE = 132_968
_LIBNNCTRL_SHA256 = "647b38c0ef77558bc58bc2b32e1003bd8c4c781eaf25e195268297edc44f4c3d"
_FUNCTION_VADDR = 0x8398
_DESCRIPTOR_SIZE = 0x480

_SOURCE_REQUIREMENTS = (
    "prt_port->data_fmt.sign = port[0]->io_desc.port_data_sign;",
    "prt_port->data_fmt.size = port[0]->io_desc.port_data_size;",
    "prt_port->data_fmt.expoffset = port[0]->io_desc.port_data_expoffset;",
    "prt_port->data_fmt.expbits = port[0]->io_desc.port_data_expbits;",
    "prt_port->dim.pitch = port[0]->io_desc.port_pitch;",
    "prt_port->dim.pitch_byte_offset = port[0]->io_desc.port_pitch_offset;",
    "prt_port->dim.pitch_bsize = port[0]->io_desc.port_pitch_bsize;",
    "prt_port->dim.dram_fmt = port[0]->io_desc.port_dram_format;",
    "prt_port->dim.bitvector = port[0]->io_desc.port_dim_bitvector;",
    "prt_port->port_size = port[0]->io_desc.port_size;",
)

# AArch64 little-endian instruction bytes from the exact recovered library.
# Each sequence includes the field load and, for packed fields, the UBFX that
# fixes the exact bit position and width.
_PROOFS = {
    "pitch": ((0x85F8, "001040b9"),),
    "pitch_byte_offset": ((0x8624, "011440b9"),),
    "pitch_bit_size": ((0x8638, "000c40f9"), (0x863C, "001440d3")),
    "dram_format": ((0x865C, "000c40f9"), (0x8660, "002446d3")),
    "bitvector": ((0x867C, "000c40f9"), (0x8680, "00284ad3")),
    "sign": ((0x85A8, "01d04139"),),
    "element_size_code": ((0x85BC, "01d44139"),),
    "exponent_offset": ((0x85D0, "01d8c139"),),
    "exponent_bits": ((0x85E4, "01dc4139"),),
    "buffer_extent": ((0x8530, "017840b9"),),
}

_FIELDS: dict[str, dict[str, int | bool]] = {
    "pitch": {"offset": 0x10, "width_bits": 32, "signed": False},
    "pitch_byte_offset": {
        "offset": 0x14,
        "width_bits": 32,
        "signed": False,
    },
    "pitch_bit_size": {
        "offset": 0x18,
        "lsb": 0,
        "width_bits": 6,
        "signed": False,
    },
    "dram_format": {
        "offset": 0x18,
        "lsb": 6,
        "width_bits": 4,
        "signed": False,
    },
    "bitvector": {
        "offset": 0x18,
        "lsb": 10,
        "width_bits": 1,
        "signed": False,
    },
    "sign": {"offset": 0x74, "width_bits": 8, "signed": False},
    "element_size_code": {
        "offset": 0x75,
        "width_bits": 8,
        "signed": False,
    },
    "exponent_offset": {
        "offset": 0x76,
        "width_bits": 8,
        "signed": True,
    },
    "exponent_bits": {
        "offset": 0x77,
        "width_bits": 8,
        "signed": False,
    },
    "buffer_extent": {
        "offset": 0x78,
        "width_bits": 32,
        "signed": False,
    },
}


def _vaddr_to_offset(elf: dict[str, Any], vaddr: int, size: int) -> int:
    for section in elf["sections"]:
        start = section["virtual_address"]
        end = start + section["size"]
        if start <= vaddr and vaddr + size <= end:
            return section["file_offset"] + (vaddr - start)
    raise DescriptorABIError(f"virtual address 0x{vaddr:x} is outside ELF sections")


def analyze_descriptor_abi(
    library: str | Path,
    source: str | Path,
) -> dict[str, Any]:
    """Verify and report the exact serialized descriptor field mapping."""

    library_path = Path(library).resolve()
    source_path = Path(source).resolve()
    try:
        raw = library_path.read_bytes()
    except OSError as exc:
        raise DescriptorABIError(f"cannot read recovered library {library_path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != _LIBNNCTRL_SIZE or digest != _LIBNNCTRL_SHA256:
        raise DescriptorABIError(
            "libnnctrl artifact mismatch: expected size "
            f"{_LIBNNCTRL_SIZE} sha256 {_LIBNNCTRL_SHA256}, observed size "
            f"{len(raw)} sha256 {digest}"
        )

    source_report = verify_source_snapshot(source_path)
    parser_path = source_path / "src/parser.c"
    try:
        parser_text = parser_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DescriptorABIError(f"cannot read exact parser source {parser_path}: {exc}") from exc
    missing_source = [item for item in _SOURCE_REQUIREMENTS if item not in parser_text]
    if missing_source:
        raise DescriptorABIError(
            f"exact parser source lacks descriptor assignments: {missing_source}"
        )

    elf = analyze_elf(library_path)
    proofs: dict[str, list[dict[str, int | str]]] = {}
    for field, instructions in _PROOFS.items():
        field_proofs: list[dict[str, int | str]] = []
        for vaddr, expected_hex in instructions:
            expected = bytes.fromhex(expected_hex)
            file_offset = _vaddr_to_offset(elf, vaddr, len(expected))
            observed = raw[file_offset : file_offset + len(expected)]
            if observed != expected:
                raise DescriptorABIError(
                    f"{field} instruction mismatch at 0x{vaddr:x}: "
                    f"expected {expected_hex}, observed {observed.hex()}"
                )
            field_proofs.append(
                {
                    "virtual_address": vaddr,
                    "file_offset": file_offset,
                    "instruction_hex": observed.hex(),
                }
            )
        proofs[field] = field_proofs

    return {
        "schema": "verkeye.cv22.descriptor-abi.v1",
        "library": {
            "path": str(library_path),
            "size": len(raw),
            "sha256": digest,
        },
        "source": {
            "path": str(source_path),
            "version": source_report["version"],
            "commit": source_report["commit"],
            "content_sha256": source_report["content_sha256"],
        },
        "function": {
            "name": "gen_net_parent_port_size",
            "vaddr": _FUNCTION_VADDR,
            "identification": (
                "source control flow and unique Abnormal sub_port_num / warning "
                "string references match recovered function"
            ),
        },
        "serialized_descriptor_size": _DESCRIPTOR_SIZE,
        "fields": _FIELDS,
        "instruction_proofs": proofs,
        "gate": {
            "status": "passed",
            "reason_codes": [],
            "proof_count": len(_FIELDS),
        },
        "claim_boundary": (
            "field offsets and packed bit slices are exact for the pinned "
            "libnnctrl artifact; numeric scale formulas and proprietary "
            "compiled-program semantics are not established by this proof"
        ),
    }
