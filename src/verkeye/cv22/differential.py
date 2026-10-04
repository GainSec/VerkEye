"""Byte-exact differential analysis for related Ambarella CV22 model images.

The analysis intentionally stops at byte identity.  A stable region across two
compiler outputs is useful for narrowing the reverse-engineering search space,
but it is not evidence that the bytes are specifically weights, instructions,
constants, or relocations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from .container import ContainerFormatError, parse_container
from .program import FOOTER_SIZE
from .schema import ArtifactSpan
from .tensors import TensorFormatError, parse_tensor_map


class DifferentialFormatError(ValueError):
    """The requested comparison cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class IdenticalRun:
    """One maximal byte-identical run at the same package-relative offset."""

    offset: int
    size: int
    sha256: str
    data: bytes

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True, slots=True)
class DifferentialArtifact:
    path: str
    size: int
    sha256: str
    package_count: int


@dataclass(frozen=True, slots=True)
class PackageMatch:
    left_split: int
    right_split: int
    left_package_span: ArtifactSpan
    right_package_span: ArtifactSpan
    compared_bytes: int
    identical_bytes: int
    coverage_of_shorter_image: float
    runs: tuple[IdenticalRun, ...]


@dataclass(frozen=True, slots=True)
class ModelDifferential:
    left: DifferentialArtifact
    right: DifferentialArtifact
    minimum_run: int
    minimum_match_bytes: int
    matches: tuple[PackageMatch, ...]
    total_identical_bytes: int
    semantic_disposition: str
    execution_semantics_proven: bool
    warning: str


SEMANTIC_DISPOSITION = "byte_identical_compiled_region"
SEMANTIC_WARNING = (
    "byte identity cannot distinguish instructions, weights, constants, "
    "relocations, or other hardware-consumed data"
)


def _digest(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def find_same_offset_runs(
    left: bytes | memoryview,
    right: bytes | memoryview,
    *,
    minimum_run: int = 64,
) -> tuple[IdenticalRun, ...]:
    """Return maximal equal runs without permitting shifted matches."""

    if minimum_run <= 0:
        raise DifferentialFormatError("minimum_run must be greater than zero")
    left_view = memoryview(left).cast("B")
    right_view = memoryview(right).cast("B")
    limit = min(len(left_view), len(right_view))
    runs: list[IdenticalRun] = []
    cursor = 0
    while cursor < limit:
        if left_view[cursor] != right_view[cursor]:
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < limit and left_view[cursor] == right_view[cursor]:
            cursor += 1
        size = cursor - start
        if size >= minimum_run:
            data = bytes(left_view[start:cursor])
            runs.append(
                IdenticalRun(
                    offset=start,
                    size=size,
                    sha256=_digest(data),
                    data=data,
                )
            )
    return tuple(runs)


@dataclass(frozen=True, slots=True)
class _ParsedArtifact:
    artifact: DifferentialArtifact
    data: bytes
    package_spans: tuple[ArtifactSpan, ...]


def _parse_artifact(path: str | Path) -> _ParsedArtifact:
    source = Path(path)
    try:
        data = source.read_bytes()
        splits = parse_tensor_map(data, parse_container(data))
    except (OSError, ContainerFormatError, TensorFormatError, ValueError) as exc:
        raise DifferentialFormatError(
            f"cannot parse CV22 artifact {source}: {exc}"
        ) from exc
    return _ParsedArtifact(
        artifact=DifferentialArtifact(
            path=str(source.resolve()),
            size=len(data),
            sha256=_digest(data),
            package_count=len(splits),
        ),
        data=data,
        package_spans=tuple(split.compiled_graph_span for split in splits),
    )


def _pair_match(
    left: _ParsedArtifact,
    right: _ParsedArtifact,
    left_index: int,
    right_index: int,
    *,
    minimum_run: int,
) -> PackageMatch:
    left_span = left.package_spans[left_index]
    right_span = right.package_spans[right_index]
    left_end = left_span.end - FOOTER_SIZE
    right_end = right_span.end - FOOTER_SIZE
    left_image = memoryview(left.data)[left_span.offset:left_end]
    right_image = memoryview(right.data)[right_span.offset:right_end]
    runs = find_same_offset_runs(left_image, right_image, minimum_run=minimum_run)
    identical = sum(run.size for run in runs)
    compared = min(len(left_image), len(right_image))
    return PackageMatch(
        left_split=left_index,
        right_split=right_index,
        left_package_span=left_span,
        right_package_span=right_span,
        compared_bytes=compared,
        identical_bytes=identical,
        coverage_of_shorter_image=(identical / compared) if compared else 0.0,
        runs=runs,
    )


def _maximum_weight_matching(
    matrix: Sequence[Sequence[PackageMatch]],
    *,
    minimum_match_bytes: int,
) -> tuple[PackageMatch, ...]:
    """Return a deterministic maximum-weight one-to-one package assignment."""

    row_count = len(matrix)
    column_count = len(matrix[0]) if row_count else 0

    @lru_cache(maxsize=None)
    def solve(row: int, used_columns: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if row == row_count:
            return 0, ()
        best_score, best_pairs = solve(row + 1, used_columns)
        for column in range(column_count):
            if used_columns & (1 << column):
                continue
            match = matrix[row][column]
            if match.identical_bytes < minimum_match_bytes:
                continue
            tail_score, tail_pairs = solve(row + 1, used_columns | (1 << column))
            candidate_score = match.identical_bytes + tail_score
            candidate_pairs = ((row, column), *tail_pairs)
            if candidate_score > best_score or (
                candidate_score == best_score and candidate_pairs < best_pairs
            ):
                best_score = candidate_score
                best_pairs = candidate_pairs
        return best_score, best_pairs

    _score, pairs = solve(0, 0)
    return tuple(matrix[row][column] for row, column in pairs)


def compare_model_variants(
    left_path: str | Path,
    right_path: str | Path,
    *,
    minimum_run: int = 64,
    minimum_match_bytes: int = 4096,
) -> ModelDifferential:
    """Compare compiled package images and retain exact artifact provenance."""

    if minimum_run <= 0:
        raise DifferentialFormatError("minimum_run must be greater than zero")
    if minimum_match_bytes < minimum_run:
        raise DifferentialFormatError(
            "minimum_match_bytes must be greater than or equal to minimum_run"
        )
    left = _parse_artifact(left_path)
    right = _parse_artifact(right_path)
    if left.artifact.sha256 == right.artifact.sha256:
        raise DifferentialFormatError("comparison requires two different artifacts")

    matrix = tuple(
        tuple(
            _pair_match(
                left,
                right,
                left_index,
                right_index,
                minimum_run=minimum_run,
            )
            for right_index in range(len(right.package_spans))
        )
        for left_index in range(len(left.package_spans))
    )
    matches = _maximum_weight_matching(
        matrix,
        minimum_match_bytes=minimum_match_bytes,
    )
    return ModelDifferential(
        left=left.artifact,
        right=right.artifact,
        minimum_run=minimum_run,
        minimum_match_bytes=minimum_match_bytes,
        matches=matches,
        total_identical_bytes=sum(match.identical_bytes for match in matches),
        semantic_disposition=SEMANTIC_DISPOSITION,
        execution_semantics_proven=False,
        warning=SEMANTIC_WARNING,
    )


def _span_document(span: ArtifactSpan) -> dict[str, int]:
    return {"offset": span.offset, "size": span.size, "end": span.end}


def inspect_model_variants(
    left_path: str | Path,
    right_path: str | Path,
    *,
    minimum_run: int = 64,
    minimum_match_bytes: int = 4096,
) -> dict[str, Any]:
    """Return deterministic JSON-ready differential evidence."""

    result = compare_model_variants(
        left_path,
        right_path,
        minimum_run=minimum_run,
        minimum_match_bytes=minimum_match_bytes,
    )

    def artifact_document(artifact: DifferentialArtifact) -> dict[str, Any]:
        return {
            "path": artifact.path,
            "size": artifact.size,
            "sha256": artifact.sha256,
            "package_count": artifact.package_count,
        }

    return {
        "schema": "verkeye.cv22.model-differential.v1",
        "artifacts": {
            "left": artifact_document(result.left),
            "right": artifact_document(result.right),
        },
        "comparison": {
            "minimum_run": result.minimum_run,
            "minimum_match_bytes": result.minimum_match_bytes,
            "matched_package_count": len(result.matches),
            "total_identical_bytes": result.total_identical_bytes,
            "semantic_disposition": result.semantic_disposition,
            "execution_semantics_proven": result.execution_semantics_proven,
            "warning": result.warning,
        },
        "matches": [
            {
                "left_split": match.left_split,
                "right_split": match.right_split,
                "left_package_span": _span_document(match.left_package_span),
                "right_package_span": _span_document(match.right_package_span),
                "compared_bytes": match.compared_bytes,
                "identical_bytes": match.identical_bytes,
                "coverage_of_shorter_image": match.coverage_of_shorter_image,
                "runs": [
                    {
                        "relative_offset": run.offset,
                        "size": run.size,
                        "end": run.end,
                        "sha256": run.sha256,
                        "left_absolute_offset": (
                            match.left_package_span.offset + run.offset
                        ),
                        "right_absolute_offset": (
                            match.right_package_span.offset + run.offset
                        ),
                    }
                    for run in match.runs
                ],
            }
            for match in result.matches
        ],
    }
