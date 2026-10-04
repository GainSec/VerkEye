"""Fail-closed, byte-complete semantic inventory for recovered CV22 DVI files.

The recovered Cavalry manifest proves the start of each DAG program relative
to its VMEM image. It does not prove boundaries within either the VMEM image
or DAG program. This inventory retains both byte ranges and stable signatures
without assigning guessed operator, parameter, or instruction meanings.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

from .program import FOOTER_CONTROL_WORD, FOOTER_MAGIC, FOOTER_SIZE


_SPLIT = re.compile(r"_split_(?P<index>\d+)\.dvi$")


class DviSemanticError(ValueError):
    """A DVI source cannot support a complete, trustworthy inventory."""


@dataclass(frozen=True, slots=True)
class DviRegion:
    kind: Literal[
        "unresolved_vmem_image",
        "unresolved_reachable_program",
        "vendor_decoded_reachable_program",
        "compiler_footer",
    ]
    offset: int
    size: int
    sha256: str
    disposition: Literal[
        "unresolved",
        "exact_structure_unknown_semantics",
        "proved_structure_unknown_semantics",
    ]
    raw_hex: str

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True, slots=True)
class DviSplitInventory:
    split_index: int
    path: str
    file_size: int
    sha256: str
    regions: tuple[DviRegion, ...]
    accounted_bytes: int
    unaccounted_bytes: int
    overlapping_spans: tuple[str, ...]
    reachable_unknown_signatures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DviSemanticInventory:
    splits: tuple[DviSplitInventory, ...]
    total_bytes: int
    accounted_bytes: int
    unresolved_computation_bytes: int
    unresolved_vmem_bytes: int
    unresolved_program_bytes: int
    decoded_program_bytes: int
    reachable_unknown_signatures: tuple[str, ...]
    executable: bool
    blocker_codes: tuple[str, ...]


def inventory_dvi_set(
    paths: Iterable[str | Path],
    *,
    dag_offsets: Mapping[int, int] | None = None,
    decoded_program_sha256: Mapping[int, str] | None = None,
) -> DviSemanticInventory:
    """Inventory DVI files without promoting opaque bytes to guessed semantics."""

    indexed: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for raw_path in paths:
        path = Path(raw_path)
        match = _SPLIT.search(path.name)
        if match is None:
            raise DviSemanticError(f"cannot determine split index from {path.name}")
        split_index = int(match.group("index"))
        if split_index in seen:
            raise DviSemanticError(f"duplicate split index: {split_index}")
        seen.add(split_index)
        indexed.append((split_index, path))
    if not indexed:
        raise DviSemanticError("DVI inventory requires at least one source")

    if dag_offsets is not None and set(dag_offsets) != seen:
        raise DviSemanticError("DAG offset set does not match DVI split set")
    if decoded_program_sha256 is not None and set(decoded_program_sha256) != seen:
        raise DviSemanticError("decoded program set does not match DVI split set")
    splits = tuple(
        _inventory_one(
            index,
            path,
            dag_offset=dag_offsets[index] if dag_offsets is not None else None,
            decoded_program_sha256=(
                decoded_program_sha256[index]
                if decoded_program_sha256 is not None
                else None
            ),
        )
        for index, path in sorted(indexed)
    )
    signatures = tuple(
        sorted(
            {
                signature
                for split in splits
                for signature in split.reachable_unknown_signatures
            }
        )
    )
    any_unresolved_program = any(
        region.kind == "unresolved_reachable_program"
        for split in splits
        for region in split.regions
    )
    blocker_codes = ["dvi_parameter_semantics_unresolved"]
    if any_unresolved_program:
        blocker_codes[:0] = [
            "dvi_program_internal_boundaries_unproven",
            "dvi_operator_semantics_unresolved",
        ]
    else:
        blocker_codes.insert(0, "dvi_operator_numeric_semantics_unproved")
    return DviSemanticInventory(
        splits=splits,
        total_bytes=sum(split.file_size for split in splits),
        accounted_bytes=sum(split.accounted_bytes for split in splits),
        unresolved_computation_bytes=sum(
            region.size
            for split in splits
            for region in split.regions
            if region.kind in {"unresolved_vmem_image", "unresolved_reachable_program"}
        ),
        unresolved_vmem_bytes=sum(
            region.size
            for split in splits
            for region in split.regions
            if region.kind == "unresolved_vmem_image"
        ),
        unresolved_program_bytes=sum(
            region.size
            for split in splits
            for region in split.regions
            if region.kind == "unresolved_reachable_program"
        ),
        decoded_program_bytes=sum(
            region.size
            for split in splits
            for region in split.regions
            if region.kind == "vendor_decoded_reachable_program"
        ),
        reachable_unknown_signatures=signatures,
        executable=False,
        blocker_codes=tuple(blocker_codes),
    )


def inventory_document(inventory: DviSemanticInventory) -> dict[str, object]:
    """Return a compact JSON document; raw bytes remain in the pinned DVI files."""

    return {
        "schema": "verkeye.cv22.dvi-semantic-inventory.v1",
        "claim_boundary": (
            "Every source byte is structurally accounted for. Cavalry image and "
            "DAG virtual addresses prove the VMEM-image/DAG-program boundary. "
            "A vendor-decoded program disposition proves operator graph structure "
            "only when its raw program digest matches independently captured ADES "
            "evidence. Numerical behavior and VMEM parameter mappings remain "
            "unproven until differential execution passes."
        ),
        "summary": {
            "split_count": len(inventory.splits),
            "total_bytes": inventory.total_bytes,
            "accounted_bytes": inventory.accounted_bytes,
            "unresolved_computation_bytes": inventory.unresolved_computation_bytes,
            "unresolved_vmem_bytes": inventory.unresolved_vmem_bytes,
            "unresolved_program_bytes": inventory.unresolved_program_bytes,
            "decoded_program_bytes": inventory.decoded_program_bytes,
            "executable": inventory.executable,
            "blocker_codes": list(inventory.blocker_codes),
            "reachable_unknown_signatures": list(
                inventory.reachable_unknown_signatures
            ),
        },
        "splits": [
            {
                "split_index": split.split_index,
                "source": {
                    "path": split.path,
                    "size": split.file_size,
                    "sha256": split.sha256,
                },
                "accounted_bytes": split.accounted_bytes,
                "unaccounted_bytes": split.unaccounted_bytes,
                "overlapping_spans": list(split.overlapping_spans),
                "reachable_unknown_signatures": list(
                    split.reachable_unknown_signatures
                ),
                "regions": [
                    {
                        "kind": region.kind,
                        "offset": region.offset,
                        "size": region.size,
                        "end": region.end,
                        "sha256": region.sha256,
                        "disposition": region.disposition,
                        "raw_bytes_preserved_in_source": True,
                    }
                    for region in split.regions
                ],
            }
            for split in inventory.splits
        ],
    }


def _inventory_one(
    split_index: int,
    path: Path,
    *,
    dag_offset: int | None,
    decoded_program_sha256: str | None,
) -> DviSplitInventory:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DviSemanticError(f"cannot read DVI source {path}: {exc}") from exc
    if len(data) <= FOOTER_SIZE:
        raise DviSemanticError(f"split {split_index} is too short for DVI footer")
    footer_offset = len(data) - FOOTER_SIZE
    footer = data[footer_offset:]
    leading, control, _raw_word, magic = struct.unpack("<IIII", footer)
    if leading != 0 or control != FOOTER_CONTROL_WORD or magic != FOOTER_MAGIC:
        raise DviSemanticError(f"split {split_index} compiler footer is invalid")

    body = data[:footer_offset]
    if dag_offset is None:
        dag_offset = 0
    if not isinstance(dag_offset, int) or not 0 < dag_offset < footer_offset:
        raise DviSemanticError(
            f"split {split_index} DAG offset {dag_offset!r} is outside the DVI body"
        )
    vmem_image = body[:dag_offset]
    dag_program = body[dag_offset:]
    program_digest = _digest(dag_program)
    if decoded_program_sha256 is not None and program_digest != decoded_program_sha256:
        raise DviSemanticError(
            f"split {split_index} decoded program SHA-256 mismatch: "
            f"expected {decoded_program_sha256}, got {program_digest}"
        )
    program_decoded = decoded_program_sha256 is not None
    regions = (
        DviRegion(
            kind="unresolved_vmem_image",
            offset=0,
            size=len(vmem_image),
            sha256=_digest(vmem_image),
            disposition="unresolved",
            raw_hex=vmem_image.hex(),
        ),
        DviRegion(
            kind=(
                "vendor_decoded_reachable_program"
                if program_decoded
                else "unresolved_reachable_program"
            ),
            offset=dag_offset,
            size=len(dag_program),
            sha256=program_digest,
            disposition=(
                "proved_structure_unknown_semantics"
                if program_decoded
                else "unresolved"
            ),
            raw_hex=dag_program.hex(),
        ),
        DviRegion(
            kind="compiler_footer",
            offset=footer_offset,
            size=FOOTER_SIZE,
            sha256=_digest(footer),
            disposition="exact_structure_unknown_semantics",
            raw_hex=footer.hex(),
        ),
    )
    overlaps = _overlaps(regions)
    accounted = sum(region.size for region in regions)
    return DviSplitInventory(
        split_index=split_index,
        path=str(path),
        file_size=len(data),
        sha256=_digest(data),
        regions=regions,
        accounted_bytes=accounted,
        unaccounted_bytes=len(data) - accounted,
        overlapping_spans=overlaps,
        reachable_unknown_signatures=tuple(
            sorted(
                region.sha256
                for region in regions
                if region.kind == "unresolved_reachable_program"
            )
        ),
    )


def _overlaps(regions: tuple[DviRegion, ...]) -> tuple[str, ...]:
    overlaps: list[str] = []
    ordered = sorted(regions, key=lambda item: (item.offset, item.end))
    for left, right in zip(ordered, ordered[1:]):
        if right.offset < left.end:
            overlaps.append(f"{left.kind}:{right.kind}")
    return tuple(overlaps)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
