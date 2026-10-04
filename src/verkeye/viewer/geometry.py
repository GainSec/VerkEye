"""Deterministic adaptation from decoded media to the CB62 model boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


TARGET_HEIGHT = 608
TARGET_WIDTH = 1088


@dataclass(frozen=True, slots=True)
class AdaptedFrame:
    """A displayable exact-boundary frame and its explicit geometry record."""

    image: NDArray[np.uint8]
    manifest: dict[str, Any]


def adapt_frame(image: NDArray[np.uint8]) -> AdaptedFrame:
    """Center-letterbox a decoded BGR frame into the 1088x608 CB62 boundary."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("viewer input must be a decoded HxWx3 image")
    if image.dtype != np.uint8:
        raise ValueError("viewer input must use uint8 samples")
    source_height, source_width = image.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("viewer input dimensions must be non-empty")

    scale = min(TARGET_WIDTH / source_width, TARGET_HEIGHT / source_height)
    resized_width = min(TARGET_WIDTH, max(1, int(round(source_width * scale))))
    resized_height = min(TARGET_HEIGHT, max(1, int(round(source_height * scale))))
    pad_left = (TARGET_WIDTH - resized_width) // 2
    pad_right = TARGET_WIDTH - resized_width - pad_left
    pad_top = (TARGET_HEIGHT - resized_height) // 2
    pad_bottom = TARGET_HEIGHT - resized_height - pad_top
    identity = source_width == TARGET_WIDTH and source_height == TARGET_HEIGHT

    if identity:
        output = image.copy(order="C")
    else:
        import cv2

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        output = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        output[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ] = resized

    return AdaptedFrame(
        image=np.ascontiguousarray(output),
        manifest={
            "schema": "verkeye.viewer-geometry.v1",
            "source_size": [source_height, source_width],
            "target_size": [TARGET_HEIGHT, TARGET_WIDTH],
            "policy": "centered-letterbox",
            "interpolation": "linear",
            "pad_bgr": [0, 0, 0],
            "scale": scale,
            "resized_size": [resized_height, resized_width],
            "padding": {
                "top": pad_top,
                "right": pad_right,
                "bottom": pad_bottom,
                "left": pad_left,
            },
            "identity": identity,
        },
    )
