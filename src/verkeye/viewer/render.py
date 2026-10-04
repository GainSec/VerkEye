"""Render exact CB62 detections and measured viewer state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..runtime.yolov6 import CB62CoreDetection
from .metrics import ViewerSnapshot


CB62_LABELS = {0: "person", 1: "vehicle", 2: "animal"}


def annotation_records(
    detections: Sequence[CB62CoreDetection],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for detection in detections:
        try:
            label = CB62_LABELS[detection.class_id]
        except KeyError as error:
            raise ValueError(
                f"unsupported CB62 class id: {detection.class_id}"
            ) from error
        records.append(
            {
                "class_id": detection.class_id,
                "label": label,
                "confidence": detection.confidence,
                "normalized_box": list(detection.normalized_box),
                "source_index": detection.source_index,
            }
        )
    return tuple(records)


def hud_lines(
    metrics: ViewerSnapshot,
    *,
    source_label: str,
    detection_count: int,
    paused: bool,
    recording: bool,
    banner: str | None = None,
) -> tuple[str, ...]:
    state = "PAUSED" if paused else "LIVE"
    rec = "REC" if recording else "NO REC"
    lines = (
        f"{source_label} | {state} | {rec} | detections {detection_count}",
        f"capture {metrics.capture_fps:.1f} fps | "
        f"inference {metrics.inference_fps:.1f} fps | "
        f"display {metrics.display_fps:.1f} fps",
        f"captured {metrics.captured_frames} | inferred {metrics.inferred_frames} | "
        f"displayed {metrics.displayed_frames} | dropped {metrics.dropped_frames} | "
        f"recorded {metrics.recorded_frames}",
        "q/esc quit | space/p pause-resume",
    )
    return ((banner,) + lines) if banner else lines


def _color(class_id: int) -> tuple[int, int, int]:
    return {
        0: (48, 220, 48),
        1: (255, 160, 32),
        2: (64, 96, 255),
    }[class_id]


def render_frame(
    image: NDArray[np.uint8],
    detections: Sequence[CB62CoreDetection],
    metrics: ViewerSnapshot,
    *,
    source_label: str,
    paused: bool,
    recording: bool,
    banner: str | None = None,
) -> NDArray[np.uint8]:
    """Draw annotations on a copy of the exact 1088x608 model-input frame."""

    import cv2

    if image.shape != (608, 1088, 3) or image.dtype != np.uint8:
        raise ValueError("render frame must be exactly 608x1088x3 uint8")
    output = image.copy(order="C")
    records = annotation_records(detections)
    for record in records:
        x1, y1, x2, y2 = record["normalized_box"]
        left = int(round(x1 * 1088))
        top = int(round(y1 * 608))
        right = int(round(x2 * 1088))
        bottom = int(round(y2 * 608))
        color = _color(int(record["class_id"]))
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        cv2.putText(
            output,
            f'{record["label"]} {record["confidence"]:.3f}',
            (left, max(18, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    lines = hud_lines(
        metrics,
        source_label=source_label,
        detection_count=len(records),
        paused=paused,
        recording=recording,
        banner=banner,
    )
    overlay = output.copy()
    overlay_height = 106 + (25 if banner else 0)
    cv2.rectangle(overlay, (0, 0), (1087, overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0.0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (12, 22 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output
