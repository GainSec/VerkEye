"""Fail-closed CV22 numeric-semantics evidence.

Exact source/binary correspondence establishes the tensor storage-format
fields.  It does not establish the numeric-value formula, zero points,
semantic floating-point encoding, or parameter formats, so those remain
explicitly unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .tensors import ProvenancedValue, SplitTensorMap


@dataclass(frozen=True, slots=True)
class TensorNumericEvidence:
    split_index: int
    role: Literal["input", "output"]
    ordinal: int
    name: str
    descriptor_offset: int
    storage_dtype: str
    element_bits: int
    signed: bool
    exponent_offset: ProvenancedValue[int]
    exponent_bits: ProvenancedValue[int]
    semantic_encoding: None = None
    scale: None = None
    zero_point: None = None


@dataclass(frozen=True, slots=True)
class QuantizationGate:
    status: Literal["passed", "blocked", "failed"]
    tensor_count: int
    recovered_scale_sets: int
    recovered_zero_point_sets: int
    recovered_parameter_dtypes: int
    recovered_tensor_storage_dtypes: int
    recovered_tensor_exponent_offsets: int
    independently_verified_layers: int
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NumericSemanticsEvidence:
    tensors: tuple[TensorNumericEvidence, ...]
    gate: QuantizationGate
    interpretation_boundary: str


def analyze_numeric_semantics(
    splits: tuple[SplitTensorMap, ...],
) -> NumericSemanticsEvidence:
    """Report proven storage fields while refusing an unproven quantizer."""

    tensors = tuple(
        TensorNumericEvidence(
            split_index=tensor.split_index,
            role=tensor.role,
            ordinal=tensor.ordinal,
            name=tensor.name.value,
            descriptor_offset=tensor.span.offset,
            storage_dtype=tensor.data_format.storage_dtype,
            element_bits=tensor.data_format.element_bits,
            signed=bool(tensor.data_format.sign.value),
            exponent_offset=tensor.data_format.exponent_offset,
            exponent_bits=tensor.data_format.exponent_bits,
        )
        for split in splits
        for tensor in (*split.inputs, *split.outputs)
    )
    return NumericSemanticsEvidence(
        tensors=tensors,
        gate=QuantizationGate(
            status="blocked",
            tensor_count=len(tensors),
            recovered_scale_sets=0,
            recovered_zero_point_sets=0,
            recovered_parameter_dtypes=0,
            recovered_tensor_storage_dtypes=len(tensors),
            recovered_tensor_exponent_offsets=len(tensors),
            independently_verified_layers=0,
            reason_codes=(
                "no_scale_records",
                "no_zero_point_records",
                "parameter_dtype_unresolved",
                "no_reference_layer_output",
            ),
        ),
        interpretation_boundary=(
            "recovered nnctrl source plus matching AArch64 field loads prove tensor "
            "storage width, signedness, exponent offset, and exponent bits; the "
            "numeric-value formula, zero point, semantic floating-point encoding, "
            "parameter boundaries, and parameter dtypes remain unresolved"
        ),
    )
