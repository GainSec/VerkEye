"""Evidence-led structure for opaque Ambarella CV22 DVI packages.

The recovered ``libnnctrl`` parser treats each compiled DVI payload as one
hardware-consumed image: it records the file position, reserves the package
size rounded to 64 bytes, and seeks across the complete image.  The only
repeated structure that can be separated without guessing is the 16-byte
compiler footer present in every package in the exact recovered model.

This module deliberately does *not* label bytes as instructions, constants,
weights, or relocations.  Those semantics are not exposed by the recovered
userspace parser and the remaining image is computation-affecting, so its
presence requires the CV22 compatibility backend.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .container import parse_container
from .schema import ArtifactSpan
from .tensors import SplitTensorMap, parse_tensor_map


FOOTER_SIZE = 16
FOOTER_CONTROL_WORD = 0x21030001
FOOTER_MAGIC = 0xFEE1900D


class ProgramFormatError(ValueError):
    """A compiled package contradicts the evidenced DVI layout."""


@dataclass(frozen=True, slots=True)
class ProgramSection:
    """A byte-exact structural section in a compiled DVI package."""

    kind: Literal["opaque_hardware_image", "compiler_footer"]
    span: ArtifactSpan
    sha256: str
    disposition: Literal["unknown", "exact_structure_unknown_semantics"]


@dataclass(frozen=True, slots=True)
class CompilerFooter:
    """The repeated four-word trailer, with semantics intentionally unnamed."""

    span: ArtifactSpan
    leading_word: int
    control_word: int
    raw_word_08: int
    magic: int
    sha256: str
    disposition: Literal["exact_structure_unknown_semantics"]


@dataclass(frozen=True, slots=True)
class VendorParserEvidence:
    """Corroborating control-flow observations from recovered libnnctrl."""

    artifact_role: Literal["libnnctrl"]
    descriptor_parser_vaddr: int
    package_allocator_vaddr: int
    package_skip_vaddr: int
    allocation_alignment: int
    userspace_decodes_package: bool
    observation: str


@dataclass(frozen=True, slots=True)
class ProgramPackage:
    """One DVI package with complete, non-overlapping source accounting."""

    split_index: int
    span: ArtifactSpan
    sha256: str
    sections: tuple[ProgramSection, ...]
    footer: CompilerFooter
    unaccounted_bytes: int


@dataclass(frozen=True, slots=True)
class ProgramGate:
    """Whether portable execution semantics have been recovered."""

    status: Literal["passed", "blocked", "failed"]
    compatibility_required: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProgramMap:
    """Complete structural map of all compiled DVI packages."""

    packages: tuple[ProgramPackage, ...]
    total_bytes: int
    accounted_bytes: int
    unaccounted_bytes: int
    opaque_computation_bytes: int
    footer_bytes: int
    vendor_parser_evidence: VendorParserEvidence
    gate: ProgramGate


PARSER_EVIDENCE = VendorParserEvidence(
    artifact_role="libnnctrl",
    descriptor_parser_vaddr=0x922C,
    package_allocator_vaddr=0x1255C,
    package_skip_vaddr=0x5540,
    allocation_alignment=64,
    userspace_decodes_package=False,
    observation=(
        "libnnctrl records the package start, reserves size rounded to 64 bytes, "
        "and seeks over the complete DVI package without decoding its interior"
    ),
)


def _digest(raw: bytes | memoryview) -> str:
    return hashlib.sha256(raw).hexdigest()


def analyze_program_packages(
    data: Any,
    splits: tuple[SplitTensorMap, ...],
) -> ProgramMap:
    """Map exact package structure and refuse unsupported internal semantics."""

    view = memoryview(data).cast("B")
    packages: list[ProgramPackage] = []
    for split in splits:
        span = split.compiled_graph_span
        if span.offset < 0 or span.size < FOOTER_SIZE or span.end > len(view):
            raise ProgramFormatError(
                f"split {split.split_index} compiled package "
                f"[{span.offset}, {span.end}) is outside {len(view)} bytes "
                f"or shorter than the {FOOTER_SIZE}-byte footer"
            )

        raw = view[span.offset : span.end]
        footer_raw = bytes(raw[-FOOTER_SIZE:])
        leading_word, control_word, raw_word_08, magic = struct.unpack(
            "<IIII", footer_raw
        )
        if leading_word != 0:
            raise ProgramFormatError(
                f"split {split.split_index} footer leading word is "
                f"0x{leading_word:08x}, expected 0"
            )
        if control_word != FOOTER_CONTROL_WORD:
            raise ProgramFormatError(
                f"split {split.split_index} footer control word is "
                f"0x{control_word:08x}, expected 0x{FOOTER_CONTROL_WORD:08x}"
            )
        if magic != FOOTER_MAGIC:
            raise ProgramFormatError(
                f"split {split.split_index} footer magic is 0x{magic:08x}, "
                f"expected 0x{FOOTER_MAGIC:08x}"
            )

        image_span = ArtifactSpan(span.offset, span.size - FOOTER_SIZE)
        footer_span = ArtifactSpan(image_span.end, FOOTER_SIZE)
        image_raw = view[image_span.offset : image_span.end]
        footer = CompilerFooter(
            span=footer_span,
            leading_word=leading_word,
            control_word=control_word,
            raw_word_08=raw_word_08,
            magic=magic,
            sha256=_digest(footer_raw),
            disposition="exact_structure_unknown_semantics",
        )
        sections = (
            ProgramSection(
                kind="opaque_hardware_image",
                span=image_span,
                sha256=_digest(image_raw),
                disposition="unknown",
            ),
            ProgramSection(
                kind="compiler_footer",
                span=footer_span,
                sha256=footer.sha256,
                disposition="exact_structure_unknown_semantics",
            ),
        )
        accounted = sum(section.span.size for section in sections)
        packages.append(
            ProgramPackage(
                split_index=split.split_index,
                span=span,
                sha256=_digest(raw),
                sections=sections,
                footer=footer,
                unaccounted_bytes=span.size - accounted,
            )
        )

    total = sum(package.span.size for package in packages)
    accounted = sum(
        section.span.size
        for package in packages
        for section in package.sections
    )
    opaque = sum(package.sections[0].span.size for package in packages)
    footer_bytes = sum(package.footer.span.size for package in packages)
    return ProgramMap(
        packages=tuple(packages),
        total_bytes=total,
        accounted_bytes=accounted,
        unaccounted_bytes=total - accounted,
        opaque_computation_bytes=opaque,
        footer_bytes=footer_bytes,
        vendor_parser_evidence=PARSER_EVIDENCE,
        gate=ProgramGate(
            status="blocked",
            compatibility_required=True,
            reason_codes=(
                "vendor_userspace_skips_hardware_image",
                "no_instruction_constant_parameter_directory",
                "opaque_computation_affecting_bytes",
                "cv22_execution_semantics_required",
            ),
        ),
    )


def _span_document(span: ArtifactSpan) -> dict[str, int]:
    return {"offset": span.offset, "size": span.size, "end": span.end}


def inspect_program_map_path(path: str | Path) -> dict[str, Any]:
    """Return a deterministic report for one exact CV22 model artifact."""

    source = Path(path)
    data = source.read_bytes()
    splits = parse_tensor_map(data, parse_container(data))
    program_map = analyze_program_packages(data, splits)
    evidence = program_map.vendor_parser_evidence
    return {
        "schema": "verkeye.cv22.program-map.v1",
        "artifact": {
            "path": str(source.resolve()),
            "size": len(data),
            "sha256": _digest(data),
        },
        "coverage": {
            "package_count": len(program_map.packages),
            "total_bytes": program_map.total_bytes,
            "accounted_bytes": program_map.accounted_bytes,
            "unaccounted_bytes": program_map.unaccounted_bytes,
            "opaque_computation_bytes": program_map.opaque_computation_bytes,
            "footer_bytes": program_map.footer_bytes,
        },
        "vendor_parser_evidence": {
            "artifact_role": evidence.artifact_role,
            "descriptor_parser_vaddr": evidence.descriptor_parser_vaddr,
            "package_allocator_vaddr": evidence.package_allocator_vaddr,
            "package_skip_vaddr": evidence.package_skip_vaddr,
            "allocation_alignment": evidence.allocation_alignment,
            "userspace_decodes_package": evidence.userspace_decodes_package,
            "observation": evidence.observation,
        },
        "execution_gate": {
            "status": program_map.gate.status,
            "compatibility_required": program_map.gate.compatibility_required,
            "reason_codes": list(program_map.gate.reason_codes),
        },
        "packages": [
            {
                "split": package.split_index,
                "span": _span_document(package.span),
                "sha256": package.sha256,
                "unaccounted_bytes": package.unaccounted_bytes,
                "sections": [
                    {
                        "kind": section.kind,
                        "span": _span_document(section.span),
                        "sha256": section.sha256,
                        "disposition": section.disposition,
                    }
                    for section in package.sections
                ],
                "footer": {
                    "span": _span_document(package.footer.span),
                    "leading_word": package.footer.leading_word,
                    "control_word": package.footer.control_word,
                    "raw_word_08": package.footer.raw_word_08,
                    "magic": package.footer.magic,
                    "sha256": package.footer.sha256,
                    "disposition": package.footer.disposition,
                },
            }
            for package in program_map.packages
        ],
    }
