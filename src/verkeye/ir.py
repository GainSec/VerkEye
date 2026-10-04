"""Provenance-preserving intermediate representation for recovered CV22 models.

The IR intentionally models the boundary graph that the recovered bytes prove.
It does not turn an opaque Ambarella compiled package into a fabricated neural
network.  Every graph object carries one or more ranges in the original source
artifact and every interpretation is labeled ``exact``, ``inferred``, or
``unknown``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .cv22.container import parse_container
from .cv22.graph import stitch_split_graph
from .cv22.parameters import analyze_parameters
from .cv22.program import analyze_program_packages
from .cv22.quantization import analyze_numeric_semantics
from .cv22.schema import ArtifactSpan
from .cv22.tensors import SplitTensorMap, TensorDescriptor, parse_tensor_map
from .cv22.weights import analyze_compiled_package


class Disposition(str, Enum):
    """Strength of the interpretation attached to source bytes."""

    EXACT = "exact"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Provenance:
    source_sha256: str
    offset: int
    size: int
    disposition: Disposition
    evidence: str

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True, slots=True)
class IRQuantization:
    scale: float | tuple[float, ...] | None
    zero_point: int | tuple[int, ...] | None
    dtype: str | None
    disposition: Disposition
    provenance: tuple[Provenance, ...]


@dataclass(frozen=True, slots=True)
class IRTensor:
    identifier: str
    name: str
    role: str
    shape: tuple[int, ...]
    pitch: int
    pitch_byte_offset: int
    pitch_bit_size: int
    dram_format: int
    bitvector: bool
    storage_dtype: str
    exponent_offset: int
    exponent_bits: int
    buffer_extent: int
    disposition: Disposition
    provenance: tuple[Provenance, ...]
    quantization: IRQuantization


AttributeValue = str | int | float | bool | None | tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class IRNode:
    identifier: str
    operator: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, AttributeValue]
    disposition: Disposition
    provenance: tuple[Provenance, ...]


@dataclass(frozen=True, slots=True)
class IREdge:
    identifier: str
    producer: str
    consumer: str
    evidence: str
    disposition: Disposition
    provenance: tuple[Provenance, ...]


@dataclass(frozen=True, slots=True)
class IRConstant:
    identifier: str
    dtype: str | None
    shape: tuple[int, ...] | None
    data_hex: str | None
    disposition: Disposition
    provenance: tuple[Provenance, ...]


@dataclass(frozen=True, slots=True)
class IROpaqueRegion:
    identifier: str
    kind: str
    sha256: str
    disposition: Disposition
    provenance: tuple[Provenance, ...]


@dataclass(frozen=True, slots=True)
class IRProgramRegion:
    """One structurally evidenced region of a compiled CV22 DVI package."""

    identifier: str
    split_index: int
    kind: str
    sha256: str
    disposition: Disposition
    provenance: tuple[Provenance, ...]


@dataclass(frozen=True, slots=True)
class IRGate:
    name: str
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IRGraph:
    nodes: tuple[IRNode, ...]
    tensors: tuple[IRTensor, ...]
    edges: tuple[IREdge, ...]
    constants: tuple[IRConstant, ...]
    opaque_regions: tuple[IROpaqueRegion, ...]
    program_regions: tuple[IRProgramRegion, ...]
    external_inputs: tuple[str, ...]
    external_outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelIR:
    schema: str
    source: SourceArtifact
    graph: IRGraph
    gates: tuple[IRGate, ...]


def _provenance(
    source_sha256: str,
    span: ArtifactSpan,
    disposition: Disposition,
    evidence: str,
) -> Provenance:
    return Provenance(
        source_sha256=source_sha256,
        offset=span.offset,
        size=span.size,
        disposition=disposition,
        evidence=evidence,
    )


def tensor_identifier(tensor: TensorDescriptor) -> str:
    return f"s{tensor.split_index}:{tensor.role}:{tensor.ordinal}"


def _tensor(source_sha256: str, tensor: TensorDescriptor) -> IRTensor:
    descriptor_provenance = _provenance(
        source_sha256,
        tensor.span,
        Disposition.EXACT,
        "complete serialized 0x480-byte boundary tensor descriptor",
    )
    return IRTensor(
        identifier=tensor_identifier(tensor),
        name=tensor.name.value,
        role=tensor.role,
        shape=tensor.shape,
        pitch=tensor.memory.pitch.value,
        pitch_byte_offset=tensor.memory.pitch_byte_offset.value,
        pitch_bit_size=tensor.memory.pitch_bit_size.value,
        dram_format=tensor.memory.dram_format.value,
        bitvector=bool(tensor.memory.bitvector.value),
        storage_dtype=tensor.data_format.storage_dtype,
        exponent_offset=tensor.data_format.exponent_offset.value,
        exponent_bits=tensor.data_format.exponent_bits.value,
        buffer_extent=tensor.buffer_extent.value,
        disposition=Disposition.INFERRED,
        provenance=(descriptor_provenance,),
        quantization=IRQuantization(
            scale=None,
            zero_point=None,
            dtype=tensor.data_format.storage_dtype,
            disposition=Disposition.UNKNOWN,
            provenance=(descriptor_provenance,),
        ),
    )


def _split_node(
    source_sha256: str,
    split: SplitTensorMap,
    *,
    directory_offset: int,
    directory_size: int,
    record_name: str,
    raw_words: tuple[int, ...],
    compiled_sha256: str,
) -> IRNode:
    attributes: Mapping[str, AttributeValue] = MappingProxyType(
        {
            "split_index": split.split_index,
            "record_name": record_name,
            "input_count": len(split.inputs),
            "output_count": len(split.outputs),
            "directory_raw_words": raw_words,
            "compiled_size": split.compiled_graph_span.size,
            "compiled_sha256": compiled_sha256,
        }
    )
    return IRNode(
        identifier=f"split:{split.split_index}",
        operator="cv22.opaque_compiled_split",
        inputs=tuple(tensor_identifier(item) for item in split.inputs),
        outputs=tuple(tensor_identifier(item) for item in split.outputs),
        attributes=attributes,
        disposition=Disposition.UNKNOWN,
        provenance=(
            _provenance(
                source_sha256,
                ArtifactSpan(directory_offset, directory_size),
                Disposition.EXACT,
                "complete fixed-width split directory record",
            ),
            _provenance(
                source_sha256,
                split.compiled_graph_span,
                Disposition.UNKNOWN,
                "bounded compiled package with unresolved operator semantics",
            ),
        ),
    )


def build_ir_path(path: str | Path) -> ModelIR:
    """Build a lossless boundary-graph IR from one exact CV22 container."""

    source_path = Path(path)
    data = source_path.read_bytes()
    source_sha256 = hashlib.sha256(data).hexdigest()
    container = parse_container(data)
    splits = parse_tensor_map(data, container)
    split_graph = stitch_split_graph(splits)
    package = analyze_compiled_package(data, splits)
    programs = analyze_program_packages(data, splits)
    parameters = analyze_parameters(data, programs)
    numeric = analyze_numeric_semantics(splits)

    tensors = tuple(
        _tensor(source_sha256, tensor)
        for split in splits
        for tensor in (*split.inputs, *split.outputs)
    )
    blocks_by_split = {block.split_index: block for block in package.blocks}
    records_by_index = {record.index: record for record in container.records}
    nodes = tuple(
        _split_node(
            source_sha256,
            split,
            directory_offset=records_by_index[split.split_index].directory_offset,
            directory_size=container.directory_record_size,
            record_name=records_by_index[split.split_index].name,
            raw_words=records_by_index[split.split_index].raw_words,
            compiled_sha256=blocks_by_split[split.split_index].sha256,
        )
        for split in splits
    )

    tensor_lookup = {tensor.identifier: tensor for tensor in tensors}
    split_lookup = {split.split_index: split for split in split_graph.splits}
    edges: list[IREdge] = []
    for edge in split_graph.edges:
        producer = tensor_identifier(
            split_lookup[edge.producer_split].outputs[edge.producer_ordinal]
        )
        consumer = tensor_identifier(
            split_lookup[edge.consumer_split].inputs[edge.consumer_ordinal]
        )
        edges.append(
            IREdge(
                identifier=f"{producer}->{consumer}",
                producer=producer,
                consumer=consumer,
                evidence=edge.evidence,
                disposition=Disposition.INFERRED,
                provenance=(
                    tensor_lookup[producer].provenance[0],
                    tensor_lookup[consumer].provenance[0],
                ),
            )
        )
    opaque_regions = tuple(
        IROpaqueRegion(
            identifier=f"split:{block.split_index}:compiled",
            kind="ambarella_nnctrl_compiled_package",
            sha256=block.sha256,
            disposition=Disposition.EXACT,
            provenance=(
                _provenance(
                    source_sha256,
                    block.span,
                    Disposition.EXACT,
                    "byte-exact opaque compiled split span",
                ),
            ),
        )
        for block in package.blocks
    )
    program_regions = tuple(
        IRProgramRegion(
            identifier=(
                f"split:{program.split_index}:program:"
                f"{'opaque' if section.kind == 'opaque_hardware_image' else 'footer'}"
            ),
            split_index=program.split_index,
            kind=section.kind,
            sha256=section.sha256,
            disposition=(
                Disposition.UNKNOWN
                if section.kind == "opaque_hardware_image"
                else Disposition.EXACT
            ),
            provenance=(
                _provenance(
                    source_sha256,
                    section.span,
                    (
                        Disposition.UNKNOWN
                        if section.kind == "opaque_hardware_image"
                        else Disposition.EXACT
                    ),
                    (
                        "computation-affecting CV22 hardware image skipped by "
                        "the recovered libnnctrl userspace parser"
                        if section.kind == "opaque_hardware_image"
                        else "repeated 16-byte CV22 compiler footer with exact "
                        "structure and unresolved field semantics"
                    ),
                ),
            ),
        )
        for program in programs.packages
        for section in program.sections
    )

    return ModelIR(
        schema="verkeye.ir.v1",
        source=SourceArtifact(
            path=str(source_path.resolve()),
            size=len(data),
            sha256=source_sha256,
        ),
        graph=IRGraph(
            nodes=nodes,
            tensors=tensors,
            edges=tuple(edges),
            constants=(),
            opaque_regions=opaque_regions,
            program_regions=program_regions,
            external_inputs=tuple(
                tensor_identifier(tensor) for tensor in split_graph.external_inputs
            ),
            external_outputs=tuple(
                tensor_identifier(tensor) for tensor in split_graph.external_outputs
            ),
        ),
        gates=(
            IRGate("container", "passed", ()),
            IRGate("tensor", split_graph.gate_status, ()),
            IRGate("program", programs.gate.status, programs.gate.reason_codes),
            IRGate(
                "parameter", parameters.gate.status, parameters.gate.reason_codes
            ),
            IRGate("weight", package.gate.status, package.gate.reason_codes),
            IRGate("quantization", numeric.gate.status, numeric.gate.reason_codes),
            IRGate(
                "onnx",
                "blocked",
                ("weight_gate_blocked", "quantization_gate_blocked"),
            ),
        ),
    )
