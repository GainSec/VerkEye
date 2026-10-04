"""Headless adapters for prerecorded live-viewer verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class HeadlessDisplay:
    """Exercise the complete viewer loop without opening a native window."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.last_frame: NDArray[np.uint8] | None = None

    def open(self, _title: str) -> None:
        self.opened = True

    def show(self, frame: NDArray[np.uint8]) -> None:
        self.last_frame = frame.copy(order="C")

    def poll_key(self, _delay_ms: int) -> int:
        return -1

    def is_open(self) -> bool:
        return self.opened and not self.closed

    def close(self) -> None:
        self.closed = True

    def write_last_frame(self, path: str | Path) -> None:
        if self.last_frame is None:
            raise RuntimeError("headless display did not receive a frame")
        import cv2

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), self.last_frame):
            raise RuntimeError(f"could not write viewer frame: {target}")


def write_deterministic_video(
    path: str | Path,
    *,
    frame_count: int,
    size: tuple[int, int],
    fps: float,
) -> None:
    """Create a small reproducible MJPEG fixture with visible frame motion."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")
    if fps <= 0:
        raise ValueError("video FPS must be positive")
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not create smoke-test video: {target}")
    try:
        y, x = np.indices((height, width), dtype=np.uint16)
        for index in range(frame_count):
            frame = np.stack(
                (
                    (x + index * 7) % 256,
                    (y + index * 11) % 256,
                    (x + y + index * 13) % 256,
                ),
                axis=2,
            ).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def resolve_smoke_video(
    *,
    requested: str | Path | None,
    generated: str | Path,
    frame_count: int,
) -> Path:
    """Use caller media unchanged or refresh the generated fixture exactly."""

    if requested is not None:
        return Path(requested)
    target = Path(generated)
    write_deterministic_video(
        target,
        frame_count=frame_count,
        size=(1280, 720),
        fps=30.0,
    )
    return target
