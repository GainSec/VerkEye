"""Fail-closed parameter accounting for recovered CV22 DVI packages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .container import parse_container
from .program import ProgramMap, analyze_program_packages
from .schema import ArtifactSpan
from .tensors import parse_tensor_map


class ParameterEvidenceError(ValueError):
    """A parameter record claims more certainty than its evidence supports."""


@dataclass(frozen=True, slots=True)
class ParameterRegion:
    """A possible or verified parameter span with explicit evidence strength."""

    identifier: str
    split_index: int
    span: ArtifactSpan
    sha256: str
    dtype: str | None
    element_count: int | None
    disposition: Literal["exact", "corroborated", "inferred", "unknown"]
    verification: Literal["verified", "candidate"]
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.verification == "verified" and self.disposition in {
            "inferred",
            "unknown",
        }:
            raise ParameterEvidenceError(
                f"{self.disposition} parameter region cannot be verified"
            )
        if self.span.size <= 0:
            raise ParameterEvidenceError("parameter span must be non-empty")
        if len(self.sha256) != 64:
            raise ParameterEvidenceError("parameter sha256 must have 64 hex digits")
        try:
            bytes.fromhex(self.sha256)
        except ValueError as exc:
            raise ParameterEvidenceError(
                "parameter sha256 must have 64 hex digits"
            ) from exc


@dataclass(frozen=True, slots=True)
class ParameterGate:
    status: Literal["passed", "blocked", "failed"]
    complete_parameter_sets: int
    independently_verified_layers: int
    compatibility_required: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParameterMap:
    regions: tuple[ParameterRegion, ...]
    verified_parameter_bytes: int
    inferred_parameter_bytes: int
    opaque_computation_bytes: int
    gate: ParameterGate


def analyze_parameters(data: Any, programs: ProgramMap) -> ParameterMap:
    """Report only evidenced parameter regions; never infer from entropy alone."""

    view = memoryview(data).cast("B")
    for package in programs.packages:
        if package.span.offset < 0 or package.span.end > len(view):
            raise ParameterEvidenceError(
                f"split {package.split_index} package exceeds parameter source"
            )
    return ParameterMap(
        regions=(),
        verified_parameter_bytes=0,
        inferred_parameter_bytes=0,
        opaque_computation_bytes=programs.opaque_computation_bytes,
        gate=ParameterGate(
            status="blocked",
            complete_parameter_sets=0,
            independently_verified_layers=0,
            compatibility_required=True,
            reason_codes=(
                "no_parameter_boundaries",
                "no_parameter_element_counts",
                "no_parameter_dtypes",
                "no_parameter_quantization",
                "opaque_hardware_images_require_cv22_execution",
            ),
        ),
    )


def inspect_parameter_map_path(path: str | Path) -> dict[str, Any]:
    """Return a deterministic fail-closed parameter report."""

    source = Path(path)
    data = source.read_bytes()
    splits = parse_tensor_map(data, parse_container(data))
    programs = analyze_program_packages(data, splits)
    parameters = analyze_parameters(data, programs)
    return {
        "schema": "verkeye.cv22.parameter-map.v1",
        "artifact": {
            "path": str(source.resolve()),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "coverage": {
            "verified_parameter_bytes": parameters.verified_parameter_bytes,
            "inferred_parameter_bytes": parameters.inferred_parameter_bytes,
            "opaque_computation_bytes": parameters.opaque_computation_bytes,
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
        "regions": [],
    }
