"""Provenance-complete types and coverage gates for CV22 execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..runtime.backend import ExecutableGraph


Disposition = Literal["exact", "inferred", "unknown"]


def _validated_sha256(value: str, field: str = "sha256") -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _validated_provenance(
    provenance: tuple["SourceSpan", ...], *, owner: str
) -> tuple["SourceSpan", ...]:
    if not provenance:
        raise ValueError(f"{owner} requires at least one proven source span")
    return tuple(provenance)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A bounded byte range in a content-addressed source artifact."""

    artifact: str
    artifact_size: int
    offset: int
    size: int
    sha256: str
    disposition: Disposition
    evidence: str

    def __post_init__(self) -> None:
        if not self.artifact:
            raise ValueError("source span artifact must not be empty")
        if not isinstance(self.artifact_size, int) or self.artifact_size <= 0:
            raise ValueError("source span artifact_size must be positive")
        if not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("source span offset must be non-negative")
        if not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("source span size must be positive")
        if self.offset + self.size > self.artifact_size:
            raise ValueError("source span exceeds artifact bounds")
        _validated_sha256(self.sha256)
        if self.disposition not in ("exact", "inferred", "unknown"):
            raise ValueError("source span disposition is invalid")
        if not self.evidence:
            raise ValueError("source span evidence must not be empty")


@dataclass(frozen=True, slots=True)
class NumericSemantics:
    """Storage, accumulation, overflow, and rounding behavior of an operation."""

    storage_dtype: str
    compute_dtype: str
    overflow: str
    rounding: str
    provenance: tuple[SourceSpan, ...]
    disposition: Disposition = "exact"

    def __post_init__(self) -> None:
        for field in ("storage_dtype", "compute_dtype"):
            try:
                normalized = np.dtype(getattr(self, field)).name
            except TypeError as exc:
                raise ValueError(f"invalid {field}") from exc
            object.__setattr__(self, field, normalized)
        if self.overflow not in ("saturate", "wrap", "trap", "undefined", "unknown"):
            raise ValueError("overflow rule is invalid")
        if self.rounding not in (
            "exact",
            "nearest_even",
            "toward_zero",
            "floor",
            "ceil",
            "unknown",
        ):
            raise ValueError("rounding rule is invalid")
        if self.disposition not in ("exact", "inferred", "unknown"):
            raise ValueError("numeric semantics disposition is invalid")
        object.__setattr__(
            self,
            "provenance",
            _validated_provenance(self.provenance, owner="numeric semantics"),
        )


@dataclass(frozen=True, slots=True)
class ExecutableParameter:
    identifier: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    provenance: tuple[SourceSpan, ...]
    disposition: Disposition = "exact"

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("parameter identifier must not be empty")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError("parameter dtype is invalid") from exc
        if any(not isinstance(value, int) or value <= 0 for value in self.shape):
            raise ValueError("parameter shape values must be positive integers")
        expected_size = int(np.prod(self.shape, dtype=np.int64)) * dtype.itemsize
        if len(self.data) != expected_size:
            raise ValueError(
                f"parameter byte count mismatch: expected {expected_size}, "
                f"got {len(self.data)}"
            )
        if self.disposition not in ("exact", "inferred", "unknown"):
            raise ValueError("parameter disposition is invalid")
        object.__setattr__(self, "dtype", dtype.name)
        object.__setattr__(self, "data", bytes(self.data))
        object.__setattr__(
            self,
            "provenance",
            _validated_provenance(self.provenance, owner="parameter"),
        )


@dataclass(frozen=True, slots=True)
class LayoutTransform:
    identifier: str
    source_layout: str
    target_layout: str
    permutation: tuple[int, ...]
    provenance: tuple[SourceSpan, ...]
    disposition: Disposition = "exact"

    def __post_init__(self) -> None:
        if not self.identifier or not self.source_layout or not self.target_layout:
            raise ValueError("layout transform names must not be empty")
        if sorted(self.permutation) != list(range(len(self.permutation))):
            raise ValueError("layout transform permutation must contain every axis once")
        if self.disposition not in ("exact", "inferred", "unknown"):
            raise ValueError("layout transform disposition is invalid")
        object.__setattr__(
            self,
            "provenance",
            _validated_provenance(self.provenance, owner="layout transform"),
        )


@dataclass(frozen=True, slots=True)
class SemanticCoverage:
    executable: bool
    blockers: tuple[str, ...]
    proved_nodes: int
    total_nodes: int
    proved_parameters: int
    total_parameters: int


def graph_coverage(graph: "ExecutableGraph") -> SemanticCoverage:
    """Return deterministic reasons why a graph may not execute yet."""

    blockers: list[str] = []
    proved_nodes = 0
    for node in graph.nodes:
        semantics = node.numeric_semantics
        semantics_known = (
            node.operator != "unknown"
            and node.disposition != "unknown"
            and semantics is not None
            and semantics.disposition != "unknown"
            and semantics.overflow != "unknown"
            and semantics.rounding != "unknown"
        )
        if not semantics_known:
            blockers.append(f"node:{node.identifier}:operator_semantics_unknown")
            continue
        if not node.provenance or any(
            span.disposition == "unknown" for span in node.provenance
        ):
            blockers.append(f"node:{node.identifier}:source_span_unproven")
            continue
        proved_nodes += 1

    proved_parameters = 0
    for parameter in graph.parameters:
        if parameter.disposition == "unknown" or any(
            span.disposition == "unknown" for span in parameter.provenance
        ):
            blockers.append(f"parameter:{parameter.identifier}:source_span_unproven")
        else:
            proved_parameters += 1

    for transform in graph.layout_transforms:
        if transform.disposition == "unknown" or any(
            span.disposition == "unknown" for span in transform.provenance
        ):
            blockers.append(f"layout:{transform.identifier}:semantics_unknown")

    ordered_blockers = tuple(sorted(blockers))
    return SemanticCoverage(
        executable=not ordered_blockers,
        blockers=ordered_blockers,
        proved_nodes=proved_nodes,
        total_nodes=len(graph.nodes),
        proved_parameters=proved_parameters,
        total_parameters=len(graph.parameters),
    )
