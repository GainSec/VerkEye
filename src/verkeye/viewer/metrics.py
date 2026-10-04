"""Measured live-viewer rates and frame accounting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ViewerSnapshot:
    capture_fps: float
    inference_fps: float
    display_fps: float
    captured_frames: int
    inferred_frames: int
    displayed_frames: int
    dropped_frames: int
    recorded_frames: int


class _EventRate:
    def __init__(self, capacity: int) -> None:
        self._timestamps: deque[int] = deque(maxlen=capacity)

    def mark(self, timestamp_ns: int) -> None:
        if timestamp_ns < 0:
            raise ValueError("event timestamp must be non-negative")
        if self._timestamps and timestamp_ns < self._timestamps[-1]:
            raise ValueError("event timestamps must be monotonic")
        self._timestamps.append(timestamp_ns)

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) * 1_000_000_000.0 / elapsed if elapsed else 0.0


class ViewerMetrics:
    """Track rates without counting dropped capture frames as inference work."""

    def __init__(self, *, rate_window: int = 120) -> None:
        if rate_window < 2:
            raise ValueError("rate_window must be at least two")
        self._source_rate = _EventRate(rate_window)
        self._display_rate = _EventRate(rate_window)
        self._inference_ns: deque[int] = deque(maxlen=rate_window)
        self._captured = 0
        self._inferred = 0
        self._displayed = 0
        self._dropped = 0
        self._recorded = 0
        self._capture_fps_override: float | None = None

    def note_source(self, timestamp_ns: int) -> None:
        self._source_rate.mark(timestamp_ns)
        self._captured += 1

    def note_inference(self, elapsed_ns: int) -> None:
        if elapsed_ns <= 0:
            raise ValueError("inference duration must be positive")
        self._inference_ns.append(elapsed_ns)
        self._inferred += 1

    def note_display(self, timestamp_ns: int) -> None:
        self._display_rate.mark(timestamp_ns)
        self._displayed += 1

    def note_recorded(self) -> None:
        self._recorded += 1

    def sync_capture_stats(self, stats: dict[str, Any]) -> None:
        captured = int(stats["captured_frames"])
        dropped = int(stats["dropped_frames"])
        elapsed_ms = float(stats["capture_elapsed_ms"])
        if captured < 0 or dropped < 0 or elapsed_ms < 0:
            raise ValueError("capture statistics must be non-negative")
        self._captured = captured
        self._dropped = dropped
        self._capture_fps_override = (
            captured * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        )

    def snapshot(self) -> ViewerSnapshot:
        inference_fps = 0.0
        if self._inference_ns:
            inference_fps = (
                len(self._inference_ns) * 1_000_000_000.0
                / sum(self._inference_ns)
            )
        return ViewerSnapshot(
            capture_fps=(
                self._capture_fps_override
                if self._capture_fps_override is not None
                else self._source_rate.fps
            ),
            inference_fps=inference_fps,
            display_fps=self._display_rate.fps,
            captured_frames=self._captured,
            inferred_frames=self._inferred,
            displayed_frames=self._displayed,
            dropped_frames=self._dropped,
            recorded_frames=self._recorded,
        )

