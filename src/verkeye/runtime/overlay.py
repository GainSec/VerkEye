"""Deterministic detection metadata and optional OpenCV overlays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .yolov6 import Detection


def overlay_metadata(
    detections: Sequence[Detection],
    *,
    labels: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    active_labels = labels or {}
    return [
        {
            "box": [float(value) for value in item.box],
            "class_id": item.class_id,
            "confidence": item.confidence,
            "label": active_labels.get(item.class_id, str(item.class_id)),
            "source_head": item.source_head,
            "source_index": item.source_index,
        }
        for item in detections
    ]


def _color(class_id: int) -> tuple[int, int, int]:
    return (
        64 + ((class_id * 67) % 192),
        64 + ((class_id * 101) % 192),
        64 + ((class_id * 149) % 192),
    )


def draw_detections(
    image: NDArray[np.uint8],
    detections: Sequence[Detection],
    *,
    labels: Mapping[int, str] | None = None,
) -> NDArray[np.uint8]:
    import cv2

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("overlay image must be HxWx3")
    output = image.copy()
    active_labels = labels or {}
    for item in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in item.box)
        color = _color(item.class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = active_labels.get(item.class_id, str(item.class_id))
        text = f"{label} {item.confidence:.3f}"
        cv2.putText(
            output,
            text,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output
