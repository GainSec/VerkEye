"""Common, provenance-preserving execution backend contracts.

The objects in this module describe executable graphs, not merely parsed
container structure.  A caller may construct one only when every operator and
tensor contract is explicit.  The recovered CB62 graph does not currently
meet that condition and is therefore rejected by the reference backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..cv22.executable_ir import (
        ExecutableParameter,
        LayoutTransform,
        NumericSemantics,
        SourceSpan,
    )


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    name: str
    version: str
    fidelity: str


@dataclass(frozen=True, slots=True)
class TensorContract:
    identifier: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("tensor identifier must not be empty")
        try:
            normalized = np.dtype(self.dtype).name
        except TypeError as exc:
            raise ValueError(f"invalid tensor dtype: {self.dtype}") from exc
        if any(not isinstance(item, int) or item < 0 for item in self.shape):
            raise ValueError("tensor shape values must be non-negative integers")
        object.__setattr__(self, "dtype", normalized)


@dataclass(frozen=True, slots=True)
class ExecutableNode:
    identifier: str
    operator: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, Any]
    provenance: tuple["SourceSpan", ...] = ()
    numeric_semantics: "NumericSemantics | None" = None
    disposition: str = "unknown"

    def __post_init__(self) -> None:
        if not self.identifier or not self.operator:
            raise ValueError("node identifier and operator must not be empty")
        if not self.outputs:
            raise ValueError("executable node must have at least one output")
        if self.disposition not in ("exact", "inferred", "unknown"):
            raise ValueError("node disposition is invalid")
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


@dataclass(frozen=True, slots=True)
class ExecutableGraph:
    identifier: str
    source_sha256: str
    nodes: tuple[ExecutableNode, ...]
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    parameters: tuple["ExecutableParameter", ...] = ()
    layout_transforms: tuple["LayoutTransform", ...] = ()

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("graph identifier must not be empty")
        digest = self.source_sha256
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError("source_sha256 must be a lowercase 64-character SHA-256")
        node_ids = [item.identifier for item in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node identifiers must be unique")
        input_ids = [item.identifier for item in self.inputs]
        output_ids = [item.identifier for item in self.outputs]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("input tensor identifiers must be unique")
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("output tensor identifiers must be unique")
        parameter_ids = [item.identifier for item in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("parameter identifiers must be unique")
        if set(input_ids) & set(parameter_ids):
            raise ValueError("input and parameter identifiers must be unique")
        available = set(input_ids) | set(parameter_ids)
        for node in self.nodes:
            missing = tuple(name for name in node.inputs if name not in available)
            if missing:
                raise ValueError(
                    f"node {node.identifier} inputs are unavailable: {', '.join(missing)}"
                )
            duplicate_outputs = tuple(name for name in node.outputs if name in available)
            if duplicate_outputs:
                raise ValueError(
                    f"node {node.identifier} outputs are not unique: "
                    + ", ".join(duplicate_outputs)
                )
            if len(node.outputs) != len(set(node.outputs)):
                raise ValueError(f"node {node.identifier} outputs must be unique")
            available.update(node.outputs)
        missing_outputs = tuple(name for name in output_ids if name not in available)
        if missing_outputs:
            raise ValueError("graph outputs are unavailable: " + ", ".join(missing_outputs))
        transform_ids = [item.identifier for item in self.layout_transforms]
        if len(transform_ids) != len(set(transform_ids)):
            raise ValueError("layout transform identifiers must be unique")


@dataclass(frozen=True, slots=True)
class IntermediateTensor:
    node: str
    tensor: str
    dtype: str
    shape: tuple[int, ...]
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BackendRun:
    backend: BackendIdentity
    graph_identifier: str
    source_sha256: str
    outputs: Mapping[str, NDArray[Any]]
    intermediates: tuple[IntermediateTensor, ...]
    timings_ns: Mapping[str, int]


class InferenceBackend(Protocol):
    """Backend interface shared by reference and compatibility execution."""

    identity: BackendIdentity

    def infer(self, inputs: Mapping[str, NDArray[Any]]) -> BackendRun:
        """Execute one complete graph while retaining raw output tensors."""
