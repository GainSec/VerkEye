"""Byte-exact mapping of opaque CV22 compiled split packages.

The recovered container exposes deterministic compiled spans, but it does not
expose an evidenced directory that separates accelerator instructions from
constants or parameters.  This module therefore maps and hashes every byte
without calling any sub-range a weight tensor.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .container import parse_container
from .parameters import analyze_parameters
from .program import analyze_program_packages
from .quantization import analyze_numeric_semantics
from .schema import ArtifactSpan
from .tensors import SplitTensorMap, parse_tensor_map


CONTROL_MARKER = bytes.fromhex("00000001000321")


class CompiledPackageError(ValueError):
    """A compiled span cannot be mapped without losing byte provenance."""


@dataclass(frozen=True, slots=True)
class ByteObservation:
    """An exact byte sequence and its absolute source span."""

    span: ArtifactSpan
    value_hex: str


@dataclass(frozen=True, slots=True)
class CompiledBlock:
    """One losslessly bounded but semantically opaque compiled split block."""

    split_index: int
    span: ArtifactSpan
    sha256: str
    shannon_entropy: float
    zero_fraction: float
    disposition: Literal["opaque_compiled_package"]
    final_word: ByteObservation
    control_marker: ByteObservation | None


@dataclass(frozen=True, slots=True)
class WeightGate:
    """Fail-closed status for exact parameter recovery."""

    status: Literal["passed", "blocked", "failed"]
    complete_parameter_sets: int
    independently_verified_layers: int
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledPackageMap:
    """Complete map of all compiled bytes plus the parameter-recovery gate."""

    blocks: tuple[CompiledBlock, ...]
    total_compiled_bytes: int
    opaque_compiled_bytes: int
    gate: WeightGate


@dataclass(frozen=True, slots=True)
class SharedCompiledRun:
    """One byte-identical run between two compiled package builds."""

    primary_split: int
    primary_relative_offset: int
    secondary_split: int
    secondary_relative_offset: int
    size: int


@dataclass(frozen=True, slots=True)
class CompiledPackageComparison:
    """Deterministic aligned-anchor comparison between two model builds."""

    chunk_size: int
    minimum_run_size: int
    runs: tuple[SharedCompiledRun, ...]
    shared_bytes: int


def _entropy(raw: bytes) -> float:
    if not raw:
        return 0.0
    total = len(raw)
    return -sum(
        (count / total) * math.log2(count / total)
        for count in Counter(raw).values()
    )


def analyze_compiled_package(
    data: Any,
    splits: tuple[SplitTensorMap, ...],
) -> CompiledPackageMap:
    """Map every compiled byte while refusing unsupported weight semantics."""

    view = memoryview(data).cast("B")
    blocks: list[CompiledBlock] = []
    for split in splits:
        span = split.compiled_graph_span
        if span.offset < 0 or span.size <= 0 or span.end > len(view):
            raise CompiledPackageError(
                f"split {split.split_index} compiled span "
                f"[{span.offset}, {span.end}) is outside {len(view)} bytes"
            )
        raw = bytes(view[span.offset : span.end])
        if len(raw) < 4:
            raise CompiledPackageError(
                f"split {split.split_index} compiled span is shorter than one word"
            )
        marker_offset = raw.rfind(CONTROL_MARKER)
        marker = (
            ByteObservation(
                ArtifactSpan(span.offset + marker_offset, len(CONTROL_MARKER)),
                CONTROL_MARKER.hex(),
            )
            if marker_offset >= 0
            else None
        )
        blocks.append(
            CompiledBlock(
                split_index=split.split_index,
                span=span,
                sha256=hashlib.sha256(raw).hexdigest(),
                shannon_entropy=_entropy(raw),
                zero_fraction=raw.count(0) / len(raw),
                disposition="opaque_compiled_package",
                final_word=ByteObservation(
                    ArtifactSpan(span.end - 4, 4), raw[-4:].hex()
                ),
                control_marker=marker,
            )
        )

    total = sum(block.span.size for block in blocks)
    return CompiledPackageMap(
        blocks=tuple(blocks),
        total_compiled_bytes=total,
        opaque_compiled_bytes=total,
        gate=WeightGate(
            status="blocked",
            complete_parameter_sets=0,
            independently_verified_layers=0,
            reason_codes=(
                "no_parameter_directory",
                "no_instruction_constant_boundary",
                "no_parameter_element_counts",
                "vendor_userspace_skips_hardware_image",
                "cv22_execution_semantics_required",
            ),
        ),
    )


def compare_compiled_packages(
    primary_data: Any,
    primary_splits: tuple[SplitTensorMap, ...],
    secondary_data: Any,
    secondary_splits: tuple[SplitTensorMap, ...],
    *,
    chunk_size: int = 16,
    minimum_run_size: int = 64,
) -> CompiledPackageComparison:
    """Find maximal byte-identical runs using unique aligned chunk anchors.

    The result only establishes byte stability between builds.  It does not
    classify a shared run as code, constants, or parameters.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if minimum_run_size < chunk_size:
        raise ValueError("minimum_run_size must be at least chunk_size")

    primary_view = memoryview(primary_data).cast("B")
    secondary_view = memoryview(secondary_data).cast("B")

    def block_bytes(
        view: memoryview, splits: tuple[SplitTensorMap, ...]
    ) -> tuple[bytes, ...]:
        result: list[bytes] = []
        for split in splits:
            span = split.compiled_graph_span
            if span.offset < 0 or span.end > len(view):
                raise CompiledPackageError(
                    f"split {split.split_index} compiled span is outside source"
                )
            result.append(bytes(view[span.offset : span.end]))
        return tuple(result)

    primary_blocks = block_bytes(primary_view, primary_splits)
    secondary_blocks = block_bytes(secondary_view, secondary_splits)

    def anchors(blocks: tuple[bytes, ...]) -> dict[bytes, list[tuple[int, int]]]:
        table: dict[bytes, list[tuple[int, int]]] = {}
        for block_index, raw in enumerate(blocks):
            for offset in range(0, len(raw) - chunk_size + 1, chunk_size):
                table.setdefault(raw[offset : offset + chunk_size], []).append(
                    (block_index, offset)
                )
        return table

    primary_anchors = anchors(primary_blocks)
    secondary_anchors = anchors(secondary_blocks)
    run_keys: set[tuple[int, int, int, int, int]] = set()
    covered: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for token, primary_locations in primary_anchors.items():
        secondary_locations = secondary_anchors.get(token, ())
        if len(primary_locations) != 1 or len(secondary_locations) != 1:
            continue
        primary_index, primary_offset = primary_locations[0]
        secondary_index, secondary_offset = secondary_locations[0]
        alignment = (primary_index, secondary_index, secondary_offset - primary_offset)
        if any(
            start <= primary_offset < end
            for start, end in covered.get(alignment, ())
        ):
            continue
        primary_raw = primary_blocks[primary_index]
        secondary_raw = secondary_blocks[secondary_index]

        before = 0
        while (
            primary_offset - before > 0
            and secondary_offset - before > 0
            and primary_raw[primary_offset - before - 1]
            == secondary_raw[secondary_offset - before - 1]
        ):
            before += 1
        after = chunk_size
        while (
            primary_offset + after < len(primary_raw)
            and secondary_offset + after < len(secondary_raw)
            and primary_raw[primary_offset + after]
            == secondary_raw[secondary_offset + after]
        ):
            after += 1

        size = before + after
        if size >= minimum_run_size:
            start = primary_offset - before
            covered.setdefault(alignment, []).append((start, start + size))
            run_keys.add(
                (
                    primary_splits[primary_index].split_index,
                    start,
                    secondary_splits[secondary_index].split_index,
                    secondary_offset - before,
                    size,
                )
            )

    runs = tuple(SharedCompiledRun(*key) for key in sorted(run_keys))
    by_split: dict[int, list[tuple[int, int]]] = {}
    for run in runs:
        by_split.setdefault(run.primary_split, []).append(
            (run.primary_relative_offset, run.primary_relative_offset + run.size)
        )
    shared_bytes = 0
    for intervals in by_split.values():
        cursor_start, cursor_end = sorted(intervals)[0]
        for start, end in sorted(intervals)[1:]:
            if start <= cursor_end:
                cursor_end = max(cursor_end, end)
            else:
                shared_bytes += cursor_end - cursor_start
                cursor_start, cursor_end = start, end
        shared_bytes += cursor_end - cursor_start

    return CompiledPackageComparison(
        chunk_size=chunk_size,
        minimum_run_size=minimum_run_size,
        runs=runs,
        shared_bytes=shared_bytes,
    )


def _span_dict(span: ArtifactSpan) -> dict[str, int]:
    return {"offset": span.offset, "size": span.size, "end": span.end}


def _observation_dict(observation: ByteObservation | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {
        "span": _span_dict(observation.span),
        "value_hex": observation.value_hex,
    }


def inspect_weight_map_path(path: str | Path) -> dict[str, Any]:
    """Return the deterministic weight-gate evidence for one CV22 artifact."""

    source = Path(path)
    data = source.read_bytes()
    splits = parse_tensor_map(data, parse_container(data))
    package = analyze_compiled_package(data, splits)
    programs = analyze_program_packages(data, splits)
    parameters = analyze_parameters(data, programs)
    numeric = analyze_numeric_semantics(splits)
    return {
        "schema": "verkeye.cv22.compiled-package.v1",
        "artifact": {
            "path": str(source.resolve()),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "coverage": {
            "compiled_block_count": len(package.blocks),
            "compiled_bytes": package.total_compiled_bytes,
            "opaque_compiled_bytes": package.opaque_compiled_bytes,
            "opaque_computation_bytes": programs.opaque_computation_bytes,
            "compiler_footer_bytes": programs.footer_bytes,
            "semantic_parameter_bytes": 0,
        },
        "program_gate": {
            "status": programs.gate.status,
            "compatibility_required": programs.gate.compatibility_required,
            "reason_codes": list(programs.gate.reason_codes),
        },
        "parameter_gate": {
            "status": parameters.gate.status,
            "complete_parameter_sets": parameters.gate.complete_parameter_sets,
            "independently_verified_layers": (
                parameters.gate.independently_verified_layers
            ),
            "compatibility_required": parameters.gate.compatibility_required,
            "reason_codes": list(parameters.gate.reason_codes),
        },
        "weight_gate": {
            "status": package.gate.status,
            "complete_parameter_sets": package.gate.complete_parameter_sets,
            "independently_verified_layers": (
                package.gate.independently_verified_layers
            ),
            "reason_codes": list(package.gate.reason_codes),
        },
        "quantization_gate": {
            "status": numeric.gate.status,
            "tensor_count": numeric.gate.tensor_count,
            "recovered_scale_sets": numeric.gate.recovered_scale_sets,
            "recovered_zero_point_sets": numeric.gate.recovered_zero_point_sets,
            "recovered_parameter_dtypes": (
                numeric.gate.recovered_parameter_dtypes
            ),
            "recovered_tensor_storage_dtypes": (
                numeric.gate.recovered_tensor_storage_dtypes
            ),
            "recovered_tensor_exponent_offsets": (
                numeric.gate.recovered_tensor_exponent_offsets
            ),
            "independently_verified_layers": (
                numeric.gate.independently_verified_layers
            ),
            "reason_codes": list(numeric.gate.reason_codes),
        },
        "tensor_numeric_evidence": [
            {
                "split": tensor.split_index,
                "role": tensor.role,
                "ordinal": tensor.ordinal,
                "name": tensor.name,
                "descriptor_offset": tensor.descriptor_offset,
                "storage_dtype": tensor.storage_dtype,
                "element_bits": tensor.element_bits,
                "signed": tensor.signed,
                "exponent_offset": {
                    "value": tensor.exponent_offset.value,
                    "offset": tensor.exponent_offset.offset,
                },
                "exponent_bits": {
                    "value": tensor.exponent_bits.value,
                    "offset": tensor.exponent_bits.offset,
                },
                "semantic_encoding": tensor.semantic_encoding,
                "scale": tensor.scale,
                "zero_point": tensor.zero_point,
            }
            for tensor in numeric.tensors
        ],
        "blocks": [
            {
                "split": block.split_index,
                "span": _span_dict(block.span),
                "sha256": block.sha256,
                "shannon_entropy": round(block.shannon_entropy, 9),
                "zero_fraction": round(block.zero_fraction, 9),
                "disposition": block.disposition,
                "final_word": _observation_dict(block.final_word),
                "control_marker": _observation_dict(block.control_marker),
            }
            for block in package.blocks
        ],
        "interpretation_boundary": (
            "compiled spans are byte-exact and fully covered; no sub-range is "
            "labeled as an accelerator instruction, constant, or parameter "
            "without the proprietary NNCtrl package decoder or equivalent "
            "compiler metadata"
        ),
    }
