"""Fail-closed stitching of CV22 split inputs and outputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from .container import parse_container
from .tensors import SplitTensorMap, TensorDescriptor, parse_tensor_map


EDGE_EVIDENCE = (
    "unique earlier output with matching name, dimensions, exact memory "
    "format, exact data format, and buffer extent"
)


@dataclass(frozen=True, slots=True)
class SplitEdge:
    producer_split: int
    producer_ordinal: int
    producer_offset: int
    consumer_split: int
    consumer_ordinal: int
    consumer_offset: int
    tensor_name: str
    evidence: str = EDGE_EVIDENCE


@dataclass(frozen=True, slots=True)
class UnresolvedInput:
    tensor: TensorDescriptor
    reason: str
    candidate_producer_splits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SplitGraph:
    splits: tuple[SplitTensorMap, ...]
    edges: tuple[SplitEdge, ...]
    external_inputs: tuple[TensorDescriptor, ...]
    external_outputs: tuple[TensorDescriptor, ...]
    unresolved_inputs: tuple[UnresolvedInput, ...]
    gate_status: Literal["passed", "blocked", "failed"]


def stitch_split_graph(splits: tuple[SplitTensorMap, ...]) -> SplitGraph:
    """Connect uniquely corroborated producer/consumer tensor records."""

    outputs_by_name: dict[str, list[TensorDescriptor]] = {}
    edges: list[SplitEdge] = []
    external_inputs: list[TensorDescriptor] = []
    unresolved: list[UnresolvedInput] = []

    for split in splits:
        for tensor in split.inputs:
            name_candidates = [
                candidate
                for candidate in outputs_by_name.get(tensor.name.value, ())
                if candidate.split_index < split.split_index
            ]
            exact_candidates = [
                candidate
                for candidate in name_candidates
                if candidate.connection_signature == tensor.connection_signature
            ]
            if len(exact_candidates) == 1:
                producer = exact_candidates[0]
                edges.append(
                    SplitEdge(
                        producer_split=producer.split_index,
                        producer_ordinal=producer.ordinal,
                        producer_offset=producer.span.offset,
                        consumer_split=tensor.split_index,
                        consumer_ordinal=tensor.ordinal,
                        consumer_offset=tensor.span.offset,
                        tensor_name=tensor.name.value,
                    )
                )
            elif len(exact_candidates) > 1:
                unresolved.append(
                    UnresolvedInput(
                        tensor=tensor,
                        reason="multiple producers matched the complete tensor signature",
                        candidate_producer_splits=tuple(
                            candidate.split_index for candidate in exact_candidates
                        ),
                    )
                )
            elif name_candidates:
                unresolved.append(
                    UnresolvedInput(
                        tensor=tensor,
                        reason="name matched but tensor signature disagreed",
                        candidate_producer_splits=tuple(
                            candidate.split_index for candidate in name_candidates
                        ),
                    )
                )
            elif split.split_index == splits[0].split_index:
                external_inputs.append(tensor)
            else:
                unresolved.append(
                    UnresolvedInput(
                        tensor=tensor,
                        reason="no earlier producer matched the tensor name",
                        candidate_producer_splits=(),
                    )
                )

        for tensor in split.outputs:
            outputs_by_name.setdefault(tensor.name.value, []).append(tensor)

    consumed = {
        (edge.producer_split, edge.producer_ordinal) for edge in edges
    }
    external_outputs = tuple(
        tensor
        for split in splits
        for tensor in split.outputs
        if (tensor.split_index, tensor.ordinal) not in consumed
    )
    gate_status: Literal["passed", "blocked", "failed"] = (
        "passed" if not unresolved and external_inputs else "blocked"
    )
    return SplitGraph(
        splits=splits,
        edges=tuple(edges),
        external_inputs=tuple(external_inputs),
        external_outputs=external_outputs,
        unresolved_inputs=tuple(unresolved),
        gate_status=gate_status,
    )


def _span_dict(offset: int, size: int) -> dict[str, int]:
    return {"offset": offset, "size": size, "end": offset + size}


def _tensor_dict(tensor: TensorDescriptor) -> dict[str, Any]:
    return {
        "role": tensor.role,
        "ordinal": tensor.ordinal,
        "span": _span_dict(tensor.span.offset, tensor.span.size),
        "raw_sha256": tensor.raw_sha256,
        "name": {
            "value": tensor.name.value,
            "offset": tensor.name.offset,
            "evidence": tensor.name.evidence,
            "confidence": tensor.name.confidence,
        },
        "dimensions": [
            {
                "value": dimension.value,
                "offset": dimension.offset,
                "confidence": dimension.confidence,
            }
            for dimension in tensor.dimensions
        ],
        "buffer_extent": {
            "value": tensor.buffer_extent.value,
            "offset": tensor.buffer_extent.offset,
            "confidence": "corroborated",
        },
        "memory": {
            "pitch": tensor.memory.pitch.value,
            "pitch_byte_offset": tensor.memory.pitch_byte_offset.value,
            "pitch_bit_size": tensor.memory.pitch_bit_size.value,
            "dram_format": tensor.memory.dram_format.value,
            "bitvector": tensor.memory.bitvector.value,
            "packed_word": tensor.memory.packed_word.value,
            "offsets": {
                "pitch": tensor.memory.pitch.offset,
                "pitch_byte_offset": tensor.memory.pitch_byte_offset.offset,
                "packed_word": tensor.memory.packed_word.offset,
            },
            "confidence": "exact",
        },
        "data_format": {
            "sign": tensor.data_format.sign.value,
            "element_size_code": tensor.data_format.element_size_code.value,
            "element_bits": tensor.data_format.element_bits,
            "storage_dtype": tensor.data_format.storage_dtype,
            "exponent_offset": tensor.data_format.exponent_offset.value,
            "exponent_bits": tensor.data_format.exponent_bits.value,
            "semantic_encoding": tensor.data_format.semantic_encoding,
            "offset": tensor.data_format.sign.offset,
            "confidence": "exact",
        },
        "raw_header_words": [item.value for item in tensor.header_words],
        "raw_header_word_offsets": [item.offset for item in tensor.header_words],
        "raw_auxiliary_words": [item.value for item in tensor.auxiliary_words],
        "raw_auxiliary_word_offsets": [
            item.offset for item in tensor.auxiliary_words
        ],
    }


def inspect_tensor_graph_path(path: str | Path) -> dict[str, Any]:
    """Return a deterministic JSON-ready tensor and split-graph evidence map."""

    source = Path(path)
    data = source.read_bytes()
    splits = parse_tensor_map(data, parse_container(data))
    graph = stitch_split_graph(splits)
    descriptor_count = sum(
        len(split.inputs) + len(split.outputs) for split in splits
    )
    return {
        "schema": "verkeye.cv22.tensor-graph.v1",
        "artifact": {
            "path": str(source.resolve()),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "gate": {
            "name": "tensor",
            "status": graph.gate_status,
            "split_count": len(splits),
            "tensor_descriptor_count": descriptor_count,
            "edge_count": len(graph.edges),
            "external_input_count": len(graph.external_inputs),
            "external_output_count": len(graph.external_outputs),
            "unresolved_input_count": len(graph.unresolved_inputs),
        },
        "interpretation_boundary": (
            "dimensions, duplicated names, buffer extents, and source/binary-"
            "proven memory and data-format fields are preserved with byte "
            "provenance; unsupported operator and numeric-value semantics "
            "remain intentionally unassigned"
        ),
        "splits": [
            {
                "index": split.split_index,
                "payload": _span_dict(
                    split.payload_span.offset, split.payload_span.size
                ),
                "descriptor_prefix": _span_dict(
                    split.descriptor_span.offset, split.descriptor_span.size
                ),
                "compiled_graph": _span_dict(
                    split.compiled_graph_span.offset,
                    split.compiled_graph_span.size,
                ),
                "inputs": [_tensor_dict(tensor) for tensor in split.inputs],
                "outputs": [_tensor_dict(tensor) for tensor in split.outputs],
            }
            for split in splits
        ],
        "edges": [
            {
                "name": edge.tensor_name,
                "producer": {
                    "split": edge.producer_split,
                    "ordinal": edge.producer_ordinal,
                    "offset": edge.producer_offset,
                },
                "consumer": {
                    "split": edge.consumer_split,
                    "ordinal": edge.consumer_ordinal,
                    "offset": edge.consumer_offset,
                },
                "evidence": edge.evidence,
            }
            for edge in graph.edges
        ],
        "external_inputs": [
            {
                "split": tensor.split_index,
                "ordinal": tensor.ordinal,
                "offset": tensor.span.offset,
                "name": tensor.name.value,
            }
            for tensor in graph.external_inputs
        ],
        "external_outputs": [
            {
                "split": tensor.split_index,
                "ordinal": tensor.ordinal,
                "offset": tensor.span.offset,
                "name": tensor.name.value,
            }
            for tensor in graph.external_outputs
        ],
        "unresolved_inputs": [
            {
                "split": item.tensor.split_index,
                "ordinal": item.tensor.ordinal,
                "offset": item.tensor.span.offset,
                "name": item.tensor.name.value,
                "reason": item.reason,
                "candidate_producer_splits": list(
                    item.candidate_producer_splits
                ),
            }
            for item in graph.unresolved_inputs
        ],
    }
