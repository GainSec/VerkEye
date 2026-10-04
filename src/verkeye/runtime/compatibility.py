"""Versioned numerical compatibility rules for exact-model runtimes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TensorTolerance:
    """The declared storage and numerical contract for one tensor."""

    dtype: str
    exact: bool = False
    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self) -> None:
        try:
            normalized = np.dtype(self.dtype).name
        except TypeError as exc:
            raise ValueError(f"unsupported dtype: {self.dtype}") from exc
        object.__setattr__(self, "dtype", normalized)
        for name, value in (("atol", self.atol), ("rtol", self.rtol)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.exact and not np.issubdtype(np.dtype(normalized), np.floating):
            raise ValueError("non-floating tensors require exact comparison")


@dataclass(frozen=True)
class TensorComparison:
    passed: bool
    compared_elements: int
    differing_elements: int
    maximum_absolute_error: float
    maximum_relative_error: float
    p50_absolute_error: float
    p95_absolute_error: float


@dataclass(frozen=True)
class DetectionTolerance:
    confidence_atol: float = 0.0
    minimum_iou: float = 1.0
    exact_count: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence_atol) or self.confidence_atol < 0:
            raise ValueError("confidence_atol must be finite and non-negative")
        if not math.isfinite(self.minimum_iou) or not 0 <= self.minimum_iou <= 1:
            raise ValueError("minimum_iou must be between zero and one")


@dataclass(frozen=True)
class DetectionComparison:
    passed: bool
    expected_count: int
    actual_count: int
    matched: int
    minimum_iou: float
    maximum_confidence_error: float


@dataclass(frozen=True)
class CompatibilityProfile:
    version: int
    model_sha256: str
    derivation_status: str
    derivation: str
    tensors: Mapping[str, TensorTolerance]
    prediction: TensorTolerance
    detections: DetectionTolerance
    evidence_sha256: Mapping[str, str]


def compare_tensor(
    expected: np.ndarray,
    actual: np.ndarray,
    tolerance: TensorTolerance,
) -> TensorComparison:
    """Compare two tensors only after their structural contract is verified."""

    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    declared_dtype = np.dtype(tolerance.dtype)
    if expected_array.dtype != declared_dtype:
        raise ValueError(
            f"expected dtype {expected_array.dtype.name} does not match "
            f"declared dtype {declared_dtype.name}"
        )
    if actual_array.dtype != expected_array.dtype:
        raise ValueError(
            f"dtype drift: expected {expected_array.dtype.name}, "
            f"got {actual_array.dtype.name}"
        )
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"shape drift: expected {expected_array.shape}, got {actual_array.shape}"
        )
    if np.issubdtype(declared_dtype, np.floating):
        if not np.all(np.isfinite(expected_array)) or not np.all(
            np.isfinite(actual_array)
        ):
            raise ValueError("non-finite tensor value")

    compared_elements = int(expected_array.size)
    if tolerance.exact:
        expected_bytes = np.ascontiguousarray(expected_array).view(np.uint8).reshape(
            compared_elements, declared_dtype.itemsize
        )
        actual_bytes = np.ascontiguousarray(actual_array).view(np.uint8).reshape(
            compared_elements, declared_dtype.itemsize
        )
        element_equal = np.all(expected_bytes == actual_bytes, axis=1)
        differing_elements = int(np.count_nonzero(~element_equal))
        if np.issubdtype(declared_dtype, np.number):
            absolute = np.abs(
                actual_array.astype(np.float64) - expected_array.astype(np.float64)
            )
        else:
            absolute = (~element_equal).astype(np.float64).reshape(expected_array.shape)
        passed = differing_elements == 0
    else:
        absolute = np.abs(
            actual_array.astype(np.float64) - expected_array.astype(np.float64)
        )
        allowed = tolerance.atol + tolerance.rtol * np.abs(
            expected_array.astype(np.float64)
        )
        equal = absolute <= allowed
        differing_elements = int(np.count_nonzero(~equal))
        passed = bool(np.all(equal))

    expected_magnitude = np.abs(expected_array.astype(np.float64))
    relative = np.zeros_like(absolute, dtype=np.float64)
    np.divide(absolute, expected_magnitude, out=relative, where=expected_magnitude != 0)
    relative[(expected_magnitude == 0) & (absolute != 0)] = np.inf
    flat_absolute = absolute.reshape(-1)
    flat_relative = relative.reshape(-1)
    return TensorComparison(
        passed=passed,
        compared_elements=compared_elements,
        differing_elements=differing_elements,
        maximum_absolute_error=_maximum(flat_absolute),
        maximum_relative_error=_maximum(flat_relative),
        p50_absolute_error=_percentile(flat_absolute, 50),
        p95_absolute_error=_percentile(flat_absolute, 95),
    )


def compare_detections(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    tolerance: DetectionTolerance,
) -> DetectionComparison:
    """Greedily match same-class detections by highest IoU."""

    expected_items = tuple(_validated_detection(item) for item in expected)
    actual_items = tuple(_validated_detection(item) for item in actual)
    unused = set(range(len(actual_items)))
    matched_ious: list[float] = []
    confidence_errors: list[float] = []

    for expected_item in expected_items:
        candidates: list[tuple[float, int, float]] = []
        for index in sorted(unused):
            actual_item = actual_items[index]
            if actual_item[0] != expected_item[0]:
                continue
            iou = _intersection_over_union(expected_item[2], actual_item[2])
            confidence_error = abs(expected_item[1] - actual_item[1])
            candidates.append((iou, index, confidence_error))
        if not candidates:
            continue
        iou, index, confidence_error = max(candidates, key=lambda item: (item[0], -item[1]))
        if iou < tolerance.minimum_iou or confidence_error > tolerance.confidence_atol:
            continue
        unused.remove(index)
        matched_ious.append(iou)
        confidence_errors.append(confidence_error)

    matched = len(matched_ious)
    count_passed = (
        len(expected_items) == len(actual_items) if tolerance.exact_count else True
    )
    passed = count_passed and matched == len(expected_items)
    return DetectionComparison(
        passed=passed,
        expected_count=len(expected_items),
        actual_count=len(actual_items),
        matched=matched,
        minimum_iou=min(matched_ious, default=1.0),
        maximum_confidence_error=max(confidence_errors, default=0.0),
    )


def load_compatibility_profile(path: str | Path) -> CompatibilityProfile:
    """Load a compatibility profile into immutable runtime objects."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    tensor_rules = {
        name: TensorTolerance(**rule) for name, rule in document["tensors"].items()
    }
    return CompatibilityProfile(
        version=int(document["version"]),
        model_sha256=str(document["model_sha256"]),
        derivation_status=str(document["derivation_status"]),
        derivation=str(document["derivation"]),
        tensors=MappingProxyType(tensor_rules),
        prediction=TensorTolerance(**document["prediction"]),
        detections=DetectionTolerance(**document["detections"]),
        evidence_sha256=MappingProxyType(dict(document["evidence_sha256"])),
    )


def _validated_detection(
    item: Mapping[str, Any],
) -> tuple[int, float, tuple[float, float, float, float]]:
    try:
        class_id = int(item["class_id"])
        confidence = float(item["confidence"])
        raw_box = item["normalized_box"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid detection contract") from exc
    if not math.isfinite(confidence):
        raise ValueError("non-finite detection confidence")
    if not isinstance(raw_box, Sequence) or len(raw_box) != 4:
        raise ValueError("normalized_box must contain four coordinates")
    box = tuple(float(value) for value in raw_box)
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in box):
        raise ValueError("normalized_box values must be finite and normalized")
    y1, x1, y2, x2 = box
    if y2 < y1 or x2 < x1:
        raise ValueError("normalized_box coordinates are inverted")
    return class_id, confidence, box


def _intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_y1, left_x1, left_y2, left_x2 = left
    right_y1, right_x1, right_y2, right_x2 = right
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection = intersection_height * intersection_width
    left_area = (left_y2 - left_y1) * (left_x2 - left_x1)
    right_area = (right_y2 - right_y1) * (right_x2 - right_x1)
    union = left_area + right_area - intersection
    return intersection / union if union else 1.0


def _maximum(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _percentile(values: np.ndarray, percentile: int) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0
