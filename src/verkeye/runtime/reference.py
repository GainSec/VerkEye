"""Inspectable exact-integer reference execution with fail-closed loading."""

from __future__ import annotations

import hashlib
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..ir import ModelIR
from .backend import (
    BackendIdentity,
    BackendRun,
    ExecutableGraph,
    IntermediateTensor,
    TensorContract,
)
from .operators import SUPPORTED_EXACT_OPERATORS, execute_exact_operator


class ReferenceExecutionError(ValueError):
    """The requested execution would exceed the recovered exact semantics."""


class ReferenceBackend:
    """Deterministic executor for graphs whose every operation is explicit."""

    identity = BackendIdentity(
        name="verkeye-reference",
        version="1",
        fidelity="exact-for-explicit-operators-only",
    )

    def __init__(self, graph: ExecutableGraph) -> None:
        self._graph = graph

    @classmethod
    def load(cls, graph: ExecutableGraph, source: bytes) -> "ReferenceBackend":
        actual = hashlib.sha256(source).hexdigest()
        if actual != graph.source_sha256:
            raise ReferenceExecutionError(
                "artifact SHA-256 mismatch: "
                f"expected {graph.source_sha256}, observed {actual}"
            )
        unsupported = sorted(
            {node.operator for node in graph.nodes} - SUPPORTED_EXACT_OPERATORS
        )
        if unsupported:
            raise ReferenceExecutionError(
                "unsupported computation-affecting operators: " + ", ".join(unsupported)
            )
        return cls(graph)

    @classmethod
    def load_model_ir(cls, model: ModelIR, source: bytes) -> "ReferenceBackend":
        actual = hashlib.sha256(source).hexdigest()
        if actual != model.source.sha256:
            raise ReferenceExecutionError(
                "artifact SHA-256 mismatch: "
                f"expected {model.source.sha256}, observed {actual}"
            )
        unsupported = sorted({node.operator for node in model.graph.nodes})
        if unsupported:
            raise ReferenceExecutionError(
                f"{', '.join(unsupported)} has unresolved computation semantics; "
                "use the CV22 compatibility backend"
            )
        raise ReferenceExecutionError("model IR contains no executable nodes")

    def infer(self, inputs: Mapping[str, NDArray[Any]]) -> BackendRun:
        started = perf_counter_ns()
        expected_names = {item.identifier for item in self._graph.inputs}
        observed_names = set(inputs)
        if observed_names != expected_names:
            missing = sorted(expected_names - observed_names)
            extra = sorted(observed_names - expected_names)
            raise ReferenceExecutionError(
                f"input set mismatch: missing={missing}, extra={extra}"
            )

        tensors: dict[str, NDArray[Any]] = {}
        for contract in self._graph.inputs:
            tensors[contract.identifier] = _validated_tensor(
                contract, inputs[contract.identifier]
            )

        intermediates: list[IntermediateTensor] = []
        node_timings: dict[str, int] = {}
        for node in self._graph.nodes:
            try:
                node_inputs = tuple(tensors[name] for name in node.inputs)
            except KeyError as exc:
                raise ReferenceExecutionError(
                    f"node {node.identifier} input is unavailable: {exc.args[0]}"
                ) from exc
            node_started = perf_counter_ns()
            try:
                result = execute_exact_operator(
                    node.operator, node_inputs, node.attributes
                )
            except ValueError as exc:
                raise ReferenceExecutionError(
                    f"node {node.identifier} failed: {exc}"
                ) from exc
            node_timings[node.identifier] = perf_counter_ns() - node_started
            if len(node.outputs) != 1:
                raise ReferenceExecutionError(
                    f"node {node.identifier} output arity is unsupported"
                )
            output_name = node.outputs[0]
            output = _read_only_copy(result)
            tensors[output_name] = output
            intermediates.append(
                IntermediateTensor(
                    node=node.identifier,
                    tensor=output_name,
                    dtype=output.dtype.name,
                    shape=tuple(output.shape),
                    size=output.nbytes,
                    sha256=hashlib.sha256(output.tobytes(order="C")).hexdigest(),
                )
            )

        outputs: dict[str, NDArray[Any]] = {}
        for contract in self._graph.outputs:
            if contract.identifier not in tensors:
                raise ReferenceExecutionError(
                    f"graph output is unavailable: {contract.identifier}"
                )
            outputs[contract.identifier] = _validated_tensor(
                contract, tensors[contract.identifier]
            )
        node_timings["total"] = perf_counter_ns() - started
        return BackendRun(
            backend=self.identity,
            graph_identifier=self._graph.identifier,
            source_sha256=self._graph.source_sha256,
            outputs=MappingProxyType(outputs),
            intermediates=tuple(intermediates),
            timings_ns=MappingProxyType(node_timings),
        )


def _validated_tensor(
    contract: TensorContract, value: NDArray[Any]
) -> NDArray[Any]:
    array = np.asarray(value)
    if array.dtype.name != contract.dtype:
        raise ReferenceExecutionError(
            f"tensor {contract.identifier} dtype mismatch: "
            f"expected {contract.dtype}, observed {array.dtype.name}"
        )
    if tuple(array.shape) != contract.shape:
        raise ReferenceExecutionError(
            f"tensor {contract.identifier} shape mismatch: "
            f"expected {contract.shape}, observed {tuple(array.shape)}"
        )
    return _read_only_copy(array)


def _read_only_copy(value: NDArray[Any]) -> NDArray[Any]:
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied
