"""Explicit, evidence-labeled preprocessing profiles and spatial transforms."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..ir import Disposition


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class EvidenceField(Generic[T]):
    value: T
    disposition: Disposition
    evidence: str


@dataclass(frozen=True, slots=True)
class PreprocessProfile:
    """All preprocessing choices; no operational field has a hidden default."""

    input_size: EvidenceField[tuple[int, int]]
    resize_mode: EvidenceField[Literal["letterbox", "stretch"]]
    source_channel_order: EvidenceField[Literal["BGR", "RGB"]]
    model_channel_order: EvidenceField[Literal["BGR", "RGB"]]
    tensor_layout: EvidenceField[Literal["NCHW"]]
    interpolation: EvidenceField[Literal["linear", "nearest"]]
    letterbox_alignment: EvidenceField[Literal["center"]]
    scale: EvidenceField[float]
    mean: EvidenceField[tuple[float, float, float]]
    std: EvidenceField[tuple[float, float, float]]
    pad_value: EvidenceField[tuple[int, int, int]]

    def to_manifest(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for field in fields(self):
            item: EvidenceField[Any] = getattr(self, field.name)
            value = list(item.value) if isinstance(item.value, tuple) else item.value
            result[field.name] = {
                "value": value,
                "disposition": item.disposition.value,
                "evidence": item.evidence,
            }
        return result


@dataclass(frozen=True, slots=True)
class SpatialTransform:
    source_width: int
    source_height: int
    input_width: int
    input_height: int
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int

    def to_source_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        converted = (
            (x1 - self.pad_left) / self.scale_x,
            (y1 - self.pad_top) / self.scale_y,
            (x2 - self.pad_left) / self.scale_x,
            (y2 - self.pad_top) / self.scale_y,
        )
        return (
            min(max(converted[0], 0.0), float(self.source_width)),
            min(max(converted[1], 0.0), float(self.source_height)),
            min(max(converted[2], 0.0), float(self.source_width)),
            min(max(converted[3], 0.0), float(self.source_height)),
        )


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    tensor: NDArray[np.float32]
    transform: SpatialTransform
    profile_manifest: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CB62ExactPreprocessResult:
    """Exact recovered model-boundary representation for the CB62 detector."""

    tensor: NDArray[np.uint8]
    transform: SpatialTransform
    profile_manifest: dict[str, Any]


_CB62_EXACT_HEIGHT = 608
_CB62_EXACT_WIDTH = 1088
_CB62_EXACT_MANIFEST: dict[str, Any] = {
    "input_size": [_CB62_EXACT_HEIGHT, _CB62_EXACT_WIDTH],
    "source_channel_order": "BGR",
    "model_channel_order": "BGR",
    "tensor_layout": "NCHW-planar-contiguous",
    "storage_dtype": "uint8",
    "normalization": "none",
    "geometry_policy": "reject-nonmatching-geometry",
    "fidelity": "exact-model-boundary",
}


def preprocess_cb62_exact_bgr(
    image: NDArray[np.uint8],
) -> CB62ExactPreprocessResult:
    """Pack an exact-size decoded BGR image into the recovered CB62 input.

    The production pipeline supplies a 1088x608 frame to a same-size target,
    then copies three uint8 B, G, and R planes into the model input.  Resizing
    is intentionally unavailable here because no non-identity production
    geometry was recovered and verified.
    """

    expected_shape = (_CB62_EXACT_HEIGHT, _CB62_EXACT_WIDTH, 3)
    if image.shape != expected_shape:
        raise ValueError("CB62 exact input must be exactly 608x1088x3 HxWxC")
    if image.dtype != np.uint8:
        raise ValueError("CB62 exact input must use uint8 samples")

    tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...])
    return CB62ExactPreprocessResult(
        tensor=tensor,
        transform=SpatialTransform(
            source_width=_CB62_EXACT_WIDTH,
            source_height=_CB62_EXACT_HEIGHT,
            input_width=_CB62_EXACT_WIDTH,
            input_height=_CB62_EXACT_HEIGHT,
            scale_x=1.0,
            scale_y=1.0,
            pad_left=0,
            pad_top=0,
        ),
        profile_manifest=dict(_CB62_EXACT_MANIFEST),
    )


def _validate_profile(profile: PreprocessProfile) -> None:
    for field in fields(profile):
        item: EvidenceField[Any] = getattr(profile, field.name)
        if item.disposition is Disposition.UNKNOWN:
            raise ValueError(f"{field.name} has unknown disposition")
        if not item.evidence.strip():
            raise ValueError(f"{field.name} has no evidence statement")
    input_height, input_width = profile.input_size.value
    if input_height <= 0 or input_width <= 0:
        raise ValueError("input_size values must be positive")
    if profile.tensor_layout.value != "NCHW":
        raise ValueError("only explicit NCHW tensor layout is supported")
    if profile.resize_mode.value not in {"letterbox", "stretch"}:
        raise ValueError(f"unsupported resize_mode: {profile.resize_mode.value}")
    if profile.source_channel_order.value not in {"BGR", "RGB"}:
        raise ValueError("source_channel_order must be BGR or RGB")
    if profile.model_channel_order.value not in {"BGR", "RGB"}:
        raise ValueError("model_channel_order must be BGR or RGB")
    if profile.interpolation.value not in {"linear", "nearest"}:
        raise ValueError("interpolation must be linear or nearest")
    if profile.letterbox_alignment.value != "center":
        raise ValueError("only explicit centered letterboxing is supported")
    if any(value == 0 for value in profile.std.value):
        raise ValueError("std values must be nonzero")
    if any(value < 0 or value > 255 for value in profile.pad_value.value):
        raise ValueError("pad_value entries must be between 0 and 255")


def preprocess_image(
    image: NDArray[np.uint8], profile: PreprocessProfile
) -> PreprocessResult:
    """Apply a completely specified profile and retain the inverse transform."""

    import cv2

    _validate_profile(profile)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("input image must be HxWx3")
    source_height, source_width = image.shape[:2]
    input_height, input_width = profile.input_size.value
    interpolation = {
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }[profile.interpolation.value]

    if profile.resize_mode.value == "letterbox":
        ratio = min(input_width / source_width, input_height / source_height)
        resized_width = max(1, int(round(source_width * ratio)))
        resized_height = max(1, int(round(source_height * ratio)))
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=interpolation
        )
        pad_left = (input_width - resized_width) // 2
        pad_top = (input_height - resized_height) // 2
        canvas = np.empty((input_height, input_width, 3), dtype=np.uint8)
        canvas[:, :] = np.asarray(profile.pad_value.value, dtype=np.uint8)
        canvas[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ] = resized
        prepared = canvas
        scale_x = resized_width / source_width
        scale_y = resized_height / source_height
    elif profile.resize_mode.value == "stretch":
        prepared = cv2.resize(
            image, (input_width, input_height), interpolation=interpolation
        )
        pad_left = 0
        pad_top = 0
        scale_x = input_width / source_width
        scale_y = input_height / source_height
    else:  # pragma: no cover - guarded by _validate_profile
        raise AssertionError("validated resize mode became unreachable")

    if profile.source_channel_order.value != profile.model_channel_order.value:
        prepared = prepared[:, :, ::-1]
    normalized = prepared.astype(np.float32) * np.float32(profile.scale.value)
    mean = np.asarray(profile.mean.value, dtype=np.float32)
    std = np.asarray(profile.std.value, dtype=np.float32)
    normalized = (normalized - mean) / std
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])
    return PreprocessResult(
        tensor=tensor,
        transform=SpatialTransform(
            source_width=source_width,
            source_height=source_height,
            input_width=input_width,
            input_height=input_height,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_left=pad_left,
            pad_top=pad_top,
        ),
        profile_manifest=profile.to_manifest(),
    )
