"""Explicit decoded-head YOLOv6 post-processing.

This module does not claim that the unresolved CB62 terminal tensors have this
layout.  Callers must provide an evidence-labeled profile before decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..ir import Disposition
from .preprocess import EvidenceField, SpatialTransform


@dataclass(frozen=True, slots=True)
class YoloV6Profile:
    tensor_layout: EvidenceField[str | None]
    confidence_threshold: EvidenceField[float]
    iou_threshold: EvidenceField[float]

    def to_manifest(self) -> dict[str, dict[str, Any]]:
        return {
            field.name: {
                "value": getattr(self, field.name).value,
                "disposition": getattr(self, field.name).disposition.value,
                "evidence": getattr(self, field.name).evidence,
            }
            for field in fields(self)
        }


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]
    source_head: int
    source_index: int


_CB62_RAW_LAYOUT = (
    "raw_x",
    "raw_y",
    "raw_log_w",
    "raw_log_h",
    "objectness",
    "class_0",
    "class_1",
    "class_2",
)


def _manifest_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_manifest_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CB62RawHeadProfile:
    """Explicit settings for the binary-proved CB62 raw-head core.

    These fields describe the exact detector-core object, whose threshold
    members are initialized by its recovered constructor. The separately
    recovered product configuration controls later routing and stateful
    cross-frame deduplication; it must not be substituted for these core
    thresholds. A caller must supply every value with evidence and unknown
    fields are rejected. The corresponding decoder is named
    ``decode_cb62_raw_head_core`` to keep that downstream boundary explicit.
    """

    input_size: EvidenceField[tuple[int, int]]
    strides: EvidenceField[tuple[int, int, int]]
    raw_layout: EvidenceField[tuple[str, ...]]
    class_thresholds: EvidenceField[tuple[float, float, float]]
    nms_distance_threshold: EvidenceField[float]
    maximum_detections: EvidenceField[int]
    minimum_normalized_area: EvidenceField[float]
    border_margin: EvidenceField[float]
    aspect_ratio_bounds: EvidenceField[tuple[float, float]]

    def to_manifest(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for field in fields(self):
            item: EvidenceField[Any] = getattr(self, field.name)
            result[field.name] = {
                "value": _manifest_value(item.value),
                "disposition": item.disposition.value,
                "evidence": item.evidence,
            }
        return result


@dataclass(frozen=True, slots=True)
class CB62CoreDetection:
    """One binary-proved raw-head detection before unresolved secondary dedup."""

    class_id: int
    confidence: float
    normalized_box: tuple[float, float, float, float]
    source_index: int
    grid: tuple[int, int]
    stride: int


def _validate_profile(profile: YoloV6Profile) -> None:
    for field in fields(profile):
        item: EvidenceField[Any] = getattr(profile, field.name)
        if item.disposition is Disposition.UNKNOWN:
            raise ValueError(f"{field.name} has unknown disposition")
        if not item.evidence.strip():
            raise ValueError(f"{field.name} has no evidence statement")
    if profile.tensor_layout.value != "cxcywh_objectness_class_scores":
        raise ValueError(f"unsupported tensor_layout: {profile.tensor_layout.value}")
    if not 0.0 <= profile.confidence_threshold.value <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")
    if not 0.0 <= profile.iou_threshold.value <= 1.0:
        raise ValueError("iou_threshold must be between zero and one")


def _validate_cb62_profile(profile: CB62RawHeadProfile) -> None:
    for field in fields(profile):
        item: EvidenceField[Any] = getattr(profile, field.name)
        if item.disposition is Disposition.UNKNOWN:
            raise ValueError(f"{field.name} has unknown disposition")
        if not item.evidence.strip():
            raise ValueError(f"{field.name} has no evidence statement")
    if profile.input_size.value != (608, 1088):
        raise ValueError("CB62 input_size must be exactly (608, 1088)")
    if profile.strides.value != (8, 16, 32):
        raise ValueError("CB62 strides must be exactly (8, 16, 32)")
    if profile.raw_layout.value != _CB62_RAW_LAYOUT:
        raise ValueError("CB62 raw_layout does not match the pinned 8-float ABI")
    if any(not 0.0 <= value <= 1.0 for value in profile.class_thresholds.value):
        raise ValueError("class_thresholds must be between zero and one")
    if not 0.0 <= profile.nms_distance_threshold.value <= 1.0:
        raise ValueError("nms_distance_threshold must be between zero and one")
    if profile.maximum_detections.value < 0:
        raise ValueError("maximum_detections must be non-negative")
    if profile.minimum_normalized_area.value < 0.0:
        raise ValueError("minimum_normalized_area must be non-negative")
    if not 0.0 <= profile.border_margin.value < 0.5:
        raise ValueError("border_margin must be in [0, 0.5)")
    minimum_aspect, maximum_aspect = profile.aspect_ratio_bounds.value
    if minimum_aspect <= 0.0 or maximum_aspect < minimum_aspect:
        raise ValueError("aspect_ratio_bounds must be positive and ordered")


@lru_cache(maxsize=16)
def build_cb62_grid(
    input_height: int,
    input_width: int,
    strides: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    """Enumerate the grid order reconstructed from ``FUN_004fd450``."""

    if input_height <= 0 or input_width <= 0:
        raise ValueError("input dimensions must be positive")
    if not strides or any(stride <= 0 for stride in strides):
        raise ValueError("strides must be positive")
    grid: list[tuple[int, int, int]] = []
    for stride in strides:
        grid_height = input_height // stride
        grid_width = input_width // stride
        for grid_y in range(grid_height):
            for grid_x in range(grid_width):
                grid.append((grid_x, grid_y, stride))
    return tuple(grid)


def cb62_iou_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Return the vendor NMS distance, exactly ``1 - IoU``."""

    return 1.0 - _iou(left, right)


def decode_cb62_raw_head_core(
    predictions: NDArray[np.floating[Any]],
    profile: CB62RawHeadProfile,
) -> tuple[CB62CoreDetection, ...]:
    """Execute the proved raw-head core, excluding unresolved secondary dedup.

    Candidate lists are built and sorted independently for class IDs 0, 1,
    and 2, matching the outer class loop in the recovered AArch64 function.
    The global accepted set is used for the binary-proved IoU-distance test.
    """

    _validate_cb62_profile(profile)
    array = np.asarray(predictions)
    if array.dtype != np.float32 or array.shape != (13_566, 8):
        raise ValueError("CB62 raw predictions must be exactly 13566x8 float32")

    grid = build_cb62_grid(
        profile.input_size.value[0],
        profile.input_size.value[1],
        profile.strides.value,
    )
    class_scores = array[:, 5:8]
    invalid_confidence_rows = np.flatnonzero(np.any(np.isnan(class_scores), axis=1))
    if invalid_confidence_rows.size:
        raise ValueError(
            "NaN confidence value returned by detection model at row "
            f"{int(invalid_confidence_rows[0])}"
        )
    class_ids = np.argmax(class_scores, axis=1)
    chosen_scores = np.take_along_axis(
        class_scores, class_ids[:, None], axis=1
    )[:, 0]
    confidences = np.asarray(array[:, 4] * chosen_scores, dtype=np.float32)
    thresholds = np.asarray(profile.class_thresholds.value, dtype=np.float32)
    eligible = confidences > thresholds[class_ids]
    candidates: list[np.ndarray] = []
    for class_id in range(3):
        source_indices = np.flatnonzero(eligible & (class_ids == class_id))
        order = np.lexsort((source_indices, -confidences[source_indices]))
        candidates.append(source_indices[order])

    accepted: list[CB62CoreDetection] = []
    input_height, input_width = profile.input_size.value
    minimum_aspect, maximum_aspect = profile.aspect_ratio_bounds.value
    for class_id in range(3):
        for source_index_raw in candidates[class_id]:
            source_index = int(source_index_raw)
            confidence = float(confidences[source_index])
            row = array[source_index]
            if np.any(np.isnan(row[:4])):
                raise ValueError(
                    f"NaN bbox coordinates returned by detection model at row {source_index}"
                )
            grid_x, grid_y, stride = grid[source_index]
            stride_f = np.float32(stride)
            center_x = np.float32(stride_f * np.float32(grid_x + row[0]))
            center_y = np.float32(stride_f * np.float32(grid_y + row[1]))
            width = np.float32(stride_f * np.exp(np.float32(row[2])))
            height = np.float32(stride_f * np.exp(np.float32(row[3])))
            half_width = np.float32(width * np.float32(0.5))
            half_height = np.float32(height * np.float32(0.5))
            x1 = np.float32(max(np.float32(0.0), center_x - half_width))
            y1 = np.float32(max(np.float32(0.0), center_y - half_height))
            x2 = np.float32(min(np.float32(input_width), center_x + half_width))
            y2 = np.float32(min(np.float32(input_height), center_y + half_height))
            normalized_box = (
                float(np.float32(x1 / np.float32(input_width))),
                float(np.float32(y1 / np.float32(input_height))),
                float(np.float32(x2 / np.float32(input_width))),
                float(np.float32(y2 / np.float32(input_height))),
            )
            normalized_width = normalized_box[2] - normalized_box[0]
            normalized_height = normalized_box[3] - normalized_box[1]
            if normalized_width <= 0.0 or normalized_height <= 0.0:
                continue
            area = normalized_width * normalized_height
            if area < profile.minimum_normalized_area.value:
                continue
            margin = profile.border_margin.value
            wholly_inside_margin = (
                margin < normalized_box[0]
                and normalized_box[2] < 1.0 - margin
                and margin < normalized_box[1]
                and normalized_box[3] < 1.0 - margin
            )
            aspect = normalized_height / normalized_width
            aspect_allowed = minimum_aspect <= aspect <= maximum_aspect
            if not wholly_inside_margin and not aspect_allowed:
                continue
            if any(
                cb62_iou_distance(item.normalized_box, normalized_box)
                < profile.nms_distance_threshold.value
                for item in accepted
            ):
                continue
            accepted.append(
                CB62CoreDetection(
                    class_id=class_id,
                    confidence=confidence,
                    normalized_box=normalized_box,
                    source_index=source_index,
                    grid=(grid_x, grid_y),
                    stride=stride,
                )
            )
            # The recovered function increments its count after insertion and
            # exits only when configured_max < count; preserve that boundary.
            if profile.maximum_detections.value < len(accepted):
                return tuple(accepted)
    return tuple(accepted)


def _heads(
    predictions: NDArray[np.floating[Any]]
    | Sequence[NDArray[np.floating[Any]]],
) -> tuple[NDArray[np.floating[Any]], ...]:
    raw_heads = (predictions,) if isinstance(predictions, np.ndarray) else predictions
    normalized: list[NDArray[np.floating[Any]]] = []
    column_count: int | None = None
    for head in raw_heads:
        array = np.asarray(head)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or array.shape[1] < 6:
            raise ValueError("each decoded prediction head must be Nx(5+C)")
        if column_count is None:
            column_count = array.shape[1]
        elif array.shape[1] != column_count:
            raise ValueError("decoded prediction heads disagree on class count")
        if not np.all(np.isfinite(array)):
            raise ValueError("decoded predictions contain non-finite values")
        normalized.append(array)
    return tuple(normalized)


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(
        0.0, right[3] - right[1]
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def decode_yolov6(
    predictions: NDArray[np.floating[Any]]
    | Sequence[NDArray[np.floating[Any]]],
    transform: SpatialTransform,
    profile: YoloV6Profile,
) -> tuple[Detection, ...]:
    """Decode explicitly identified post-head predictions and apply class NMS."""

    _validate_profile(profile)
    candidates: list[Detection] = []
    for head_index, head in enumerate(_heads(predictions)):
        for source_index, row in enumerate(head):
            class_id = int(np.argmax(row[5:]))
            confidence = float(row[4] * row[5 + class_id])
            if confidence < profile.confidence_threshold.value:
                continue
            center_x, center_y, width, height = (float(value) for value in row[:4])
            box = transform.to_source_box(
                (
                    center_x - width / 2.0,
                    center_y - height / 2.0,
                    center_x + width / 2.0,
                    center_y + height / 2.0,
                )
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            candidates.append(
                Detection(class_id, confidence, box, head_index, source_index)
            )

    candidates.sort(
        key=lambda item: (
            -item.confidence,
            item.class_id,
            item.source_head,
            item.source_index,
        )
    )
    kept: list[Detection] = []
    for candidate in candidates:
        if any(
            existing.class_id == candidate.class_id
            and _iou(existing.box, candidate.box) > profile.iou_threshold.value
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return tuple(kept)
