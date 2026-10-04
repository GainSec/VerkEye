"""Deterministic JSON encoding and schema validation for VerkEye IR v1."""

from __future__ import annotations

import json
import sysconfig
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .ir import (
    AttributeValue,
    Disposition,
    IRConstant,
    IREdge,
    IRGate,
    IRGraph,
    IRNode,
    IROpaqueRegion,
    IRProgramRegion,
    IRQuantization,
    IRTensor,
    ModelIR,
    Provenance,
    SourceArtifact,
)


SCHEMA_FILENAME = "verkeye-ir-v1.schema.json"


def _schema_path() -> Path:
    candidates = (
        Path(__file__).resolve().parents[2] / "schemas" / SCHEMA_FILENAME,
        Path(sysconfig.get_path("data"))
        / "share"
        / "verkeye"
        / "schemas"
        / SCHEMA_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not locate {SCHEMA_FILENAME}; checked {checked}")


def load_schema(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else _schema_path()
    return json.loads(source.read_text())


def _provenance_document(item: Provenance) -> dict[str, Any]:
    return {
        "source_sha256": item.source_sha256,
        "offset": item.offset,
        "size": item.size,
        "end": item.end,
        "disposition": item.disposition.value,
        "evidence": item.evidence,
    }


def _quantization_document(item: IRQuantization) -> dict[str, Any]:
    return {
        "scale": item.scale,
        "zero_point": item.zero_point,
        "dtype": item.dtype,
        "disposition": item.disposition.value,
        "provenance": [_provenance_document(value) for value in item.provenance],
    }


def _attribute_document(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_attribute_document(item) for item in value]
    return value


def ir_to_document(ir: ModelIR) -> dict[str, Any]:
    """Convert immutable IR objects to a JSON-ready canonical document."""

    return {
        "schema": ir.schema,
        "source": {
            "path": ir.source.path,
            "size": ir.source.size,
            "sha256": ir.source.sha256,
        },
        "graph": {
            "nodes": [
                {
                    "identifier": item.identifier,
                    "operator": item.operator,
                    "inputs": list(item.inputs),
                    "outputs": list(item.outputs),
                    "attributes": {
                        key: _attribute_document(value)
                        for key, value in item.attributes.items()
                    },
                    "disposition": item.disposition.value,
                    "provenance": [
                        _provenance_document(value) for value in item.provenance
                    ],
                }
                for item in ir.graph.nodes
            ],
            "tensors": [
                {
                    "identifier": item.identifier,
                    "name": item.name,
                    "role": item.role,
                    "shape": list(item.shape),
                    "pitch": item.pitch,
                    "pitch_byte_offset": item.pitch_byte_offset,
                    "pitch_bit_size": item.pitch_bit_size,
                    "dram_format": item.dram_format,
                    "bitvector": item.bitvector,
                    "storage_dtype": item.storage_dtype,
                    "exponent_offset": item.exponent_offset,
                    "exponent_bits": item.exponent_bits,
                    "buffer_extent": item.buffer_extent,
                    "disposition": item.disposition.value,
                    "provenance": [
                        _provenance_document(value) for value in item.provenance
                    ],
                    "quantization": _quantization_document(item.quantization),
                }
                for item in ir.graph.tensors
            ],
            "edges": [
                {
                    "identifier": item.identifier,
                    "producer": item.producer,
                    "consumer": item.consumer,
                    "evidence": item.evidence,
                    "disposition": item.disposition.value,
                    "provenance": [
                        _provenance_document(value) for value in item.provenance
                    ],
                }
                for item in ir.graph.edges
            ],
            "constants": [
                {
                    "identifier": item.identifier,
                    "dtype": item.dtype,
                    "shape": list(item.shape) if item.shape is not None else None,
                    "data_hex": item.data_hex,
                    "disposition": item.disposition.value,
                    "provenance": [
                        _provenance_document(value) for value in item.provenance
                    ],
                }
                for item in ir.graph.constants
            ],
            "opaque_regions": [
                {
                    "identifier": item.identifier,
                    "kind": item.kind,
                    "sha256": item.sha256,
                    "disposition": item.disposition.value,
                    "provenance": [
                        _provenance_document(value) for value in item.provenance
                    ],
                }
                for item in ir.graph.opaque_regions
            ],
            "program_regions": [
                {
                    "identifier": item.identifier,
                    "split_index": item.split_index,
                    "kind": item.kind,
                    "sha256": item.sha256,
                    "disposition": item.disposition.value,
                    "provenance": [
                        _provenance_document(value) for value in item.provenance
                    ],
                }
                for item in ir.graph.program_regions
            ],
            "external_inputs": list(ir.graph.external_inputs),
            "external_outputs": list(ir.graph.external_outputs),
        },
        "gates": [
            {
                "name": item.name,
                "status": item.status,
                "reason_codes": list(item.reason_codes),
            }
            for item in ir.gates
        ],
    }


def dumps_ir(ir: ModelIR) -> str:
    """Return deterministic UTF-8 JSON with one trailing newline."""

    document = ir_to_document(ir)
    validate_ir_document(document)
    return json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ": "),
    ) + "\n"


def validate_ir_document(
    document: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
) -> None:
    """Reject any JSON document outside the strict v1 contract."""

    active_schema = dict(schema) if schema is not None else load_schema()
    validator = Draft202012Validator(active_schema)
    errors = sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(
            f"IR schema validation failed at {location}: {error.message}"
        ) from error

    _validate_semantics(document)


def _validate_semantics(document: Mapping[str, Any]) -> None:
    source = document["source"]
    source_sha256 = source["sha256"]
    source_size = source["size"]
    graph = document["graph"]
    for collection in (
        "nodes",
        "tensors",
        "edges",
        "constants",
        "opaque_regions",
        "program_regions",
    ):
        for item in graph[collection]:
            for provenance in item["provenance"]:
                _validate_provenance(provenance, source_sha256, source_size)
            if collection == "tensors":
                for provenance in item["quantization"]["provenance"]:
                    _validate_provenance(provenance, source_sha256, source_size)

    tensor_ids = [item["identifier"] for item in graph["tensors"]]
    if len(tensor_ids) != len(set(tensor_ids)):
        raise ValueError("IR semantic validation failed: duplicate tensor identifier")
    known_tensors = set(tensor_ids)
    referenced = set(graph["external_inputs"]) | set(graph["external_outputs"])
    for node in graph["nodes"]:
        referenced.update(node["inputs"])
        referenced.update(node["outputs"])
    for edge in graph["edges"]:
        referenced.add(edge["producer"])
        referenced.add(edge["consumer"])
    missing = sorted(referenced - known_tensors)
    if missing:
        raise ValueError(
            f"IR semantic validation failed: unknown tensor references {missing}"
        )


def _validate_provenance(
    provenance: Mapping[str, Any], source_sha256: str, source_size: int
) -> None:
    if provenance["source_sha256"] != source_sha256:
        raise ValueError("IR semantic validation failed: provenance source mismatch")
    if provenance["end"] != provenance["offset"] + provenance["size"]:
        raise ValueError("IR semantic validation failed: provenance end mismatch")
    if provenance["end"] > source_size:
        raise ValueError("IR semantic validation failed: provenance exceeds source")


def _provenance_from_document(item: Mapping[str, Any]) -> Provenance:
    return Provenance(
        source_sha256=item["source_sha256"],
        offset=item["offset"],
        size=item["size"],
        disposition=Disposition(item["disposition"]),
        evidence=item["evidence"],
    )


def _provenance_tuple(items: list[Mapping[str, Any]]) -> tuple[Provenance, ...]:
    return tuple(_provenance_from_document(item) for item in items)


def _freeze_attribute(value: Any) -> AttributeValue:
    if isinstance(value, list):
        return tuple(_freeze_attribute(item) for item in value)
    return value


def document_to_ir(document: Mapping[str, Any]) -> ModelIR:
    """Validate and reconstruct immutable IR objects from parsed JSON."""

    validate_ir_document(document)
    source = document["source"]
    graph = document["graph"]
    return ModelIR(
        schema=document["schema"],
        source=SourceArtifact(
            path=source["path"],
            size=source["size"],
            sha256=source["sha256"],
        ),
        graph=IRGraph(
            nodes=tuple(
                IRNode(
                    identifier=item["identifier"],
                    operator=item["operator"],
                    inputs=tuple(item["inputs"]),
                    outputs=tuple(item["outputs"]),
                    attributes=MappingProxyType(
                        {
                            key: _freeze_attribute(value)
                            for key, value in item["attributes"].items()
                        }
                    ),
                    disposition=Disposition(item["disposition"]),
                    provenance=_provenance_tuple(item["provenance"]),
                )
                for item in graph["nodes"]
            ),
            tensors=tuple(
                IRTensor(
                    identifier=item["identifier"],
                    name=item["name"],
                    role=item["role"],
                    shape=tuple(item["shape"]),
                    pitch=item["pitch"],
                    pitch_byte_offset=item["pitch_byte_offset"],
                    pitch_bit_size=item["pitch_bit_size"],
                    dram_format=item["dram_format"],
                    bitvector=item["bitvector"],
                    storage_dtype=item["storage_dtype"],
                    exponent_offset=item["exponent_offset"],
                    exponent_bits=item["exponent_bits"],
                    buffer_extent=item["buffer_extent"],
                    disposition=Disposition(item["disposition"]),
                    provenance=_provenance_tuple(item["provenance"]),
                    quantization=IRQuantization(
                        scale=_freeze_attribute(item["quantization"]["scale"]),
                        zero_point=_freeze_attribute(
                            item["quantization"]["zero_point"]
                        ),
                        dtype=item["quantization"]["dtype"],
                        disposition=Disposition(
                            item["quantization"]["disposition"]
                        ),
                        provenance=_provenance_tuple(
                            item["quantization"]["provenance"]
                        ),
                    ),
                )
                for item in graph["tensors"]
            ),
            edges=tuple(
                IREdge(
                    identifier=item["identifier"],
                    producer=item["producer"],
                    consumer=item["consumer"],
                    evidence=item["evidence"],
                    disposition=Disposition(item["disposition"]),
                    provenance=_provenance_tuple(item["provenance"]),
                )
                for item in graph["edges"]
            ),
            constants=tuple(
                IRConstant(
                    identifier=item["identifier"],
                    dtype=item["dtype"],
                    shape=(tuple(item["shape"]) if item["shape"] is not None else None),
                    data_hex=item["data_hex"],
                    disposition=Disposition(item["disposition"]),
                    provenance=_provenance_tuple(item["provenance"]),
                )
                for item in graph["constants"]
            ),
            opaque_regions=tuple(
                IROpaqueRegion(
                    identifier=item["identifier"],
                    kind=item["kind"],
                    sha256=item["sha256"],
                    disposition=Disposition(item["disposition"]),
                    provenance=_provenance_tuple(item["provenance"]),
                )
                for item in graph["opaque_regions"]
            ),
            program_regions=tuple(
                IRProgramRegion(
                    identifier=item["identifier"],
                    split_index=item["split_index"],
                    kind=item["kind"],
                    sha256=item["sha256"],
                    disposition=Disposition(item["disposition"]),
                    provenance=_provenance_tuple(item["provenance"]),
                )
                for item in graph["program_regions"]
            ),
            external_inputs=tuple(graph["external_inputs"]),
            external_outputs=tuple(graph["external_outputs"]),
        ),
        gates=tuple(
            IRGate(
                name=item["name"],
                status=item["status"],
                reason_codes=tuple(item["reason_codes"]),
            )
            for item in document["gates"]
        ),
    )
