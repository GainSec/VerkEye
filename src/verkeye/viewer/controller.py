"""Resource-safe interactive controller for exact CB62 live inference."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from ..evidence import atomic_write_json
from ..runtime.inference import ExactLiveFrameResult
from ..runtime.sources import MediaFrame
from .geometry import adapt_frame
from .metrics import ViewerMetrics
from .render import annotation_records, render_frame


class ViewerError(RuntimeError):
    """The native viewer or recording surface is unavailable."""


class ExactFrameRunner(Protocol):
    def infer_live_frame(
        self, frame: MediaFrame, *, run_id: str
    ) -> ExactLiveFrameResult: ...


class ViewerDisplay(Protocol):
    def open(self, title: str) -> None: ...

    def show(self, frame: NDArray[np.uint8]) -> None: ...

    def poll_key(self, delay_ms: int) -> int: ...

    def is_open(self) -> bool: ...

    def close(self) -> None: ...


class ViewerWriter(Protocol):
    def write(self, frame: NDArray[np.uint8]) -> None: ...

    def release(self) -> None: ...


class OpenCvDisplay:
    """OpenCV HighGUI adapter kept outside the testable controller core."""

    def __init__(self, *, title: str = "VerkEye — exact CB62 live viewer") -> None:
        self.title = title
        self._opened = False

    def open(self, title: str) -> None:
        import cv2

        self.title = title
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, 1088, 608)
        except cv2.error as error:
            raise ViewerError(
                "OpenCV GUI is unavailable; install VerkEye's viewer extra "
                "and run from a desktop session"
            ) from error
        self._opened = True

    def show(self, frame: NDArray[np.uint8]) -> None:
        import cv2

        cv2.imshow(self.title, frame)

    def poll_key(self, delay_ms: int) -> int:
        import cv2

        return cv2.waitKey(delay_ms) & 0xFF

    def is_open(self) -> bool:
        if not self._opened:
            return False
        import cv2

        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        if not self._opened:
            return
        import cv2

        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        except cv2.error:
            pass
        self._opened = False


def open_video_writer(
    path: str | Path,
    *,
    fps: float,
    codec: str = "mp4v",
) -> ViewerWriter:
    """Open an annotated 1088x608 writer or fail before inference begins."""

    if fps <= 0:
        raise ValueError("recording FPS must be positive")
    if len(codec) != 4:
        raise ValueError("recording codec must contain four characters")
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (1088, 608),
    )
    if not writer.isOpened():
        writer.release()
        raise ViewerError(f"could not open annotated recording: {target}")
    return writer


class LiveViewerSession:
    """Run one interactive exact-model viewing session."""

    def __init__(
        self,
        *,
        source: Iterable[MediaFrame],
        exact_session: ExactFrameRunner,
        display: ViewerDisplay,
        source_label: str,
        banner: str | None = None,
        run_id: str,
        writer: ViewerWriter | None = None,
        recording: dict[str, Any] | None = None,
        summary_path: str | Path | None = None,
        max_frames: int | None = None,
        window_title: str = "VerkEye — exact CB62 live viewer",
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if writer is None and recording is not None:
            raise ValueError("recording metadata requires a writer")
        if writer is not None and recording is None:
            raise ValueError("writer requires recording metadata")
        self.source = source
        self.exact_session = exact_session
        self.display = display
        self.source_label = source_label
        self.banner = banner
        self.run_id = run_id
        self.writer = writer
        self.recording = dict(recording) if recording is not None else None
        self.summary_path = Path(summary_path) if summary_path is not None else None
        self.max_frames = max_frames
        self.window_title = window_title
        self.clock_ns = clock_ns
        self.metrics = ViewerMetrics()
        self._geometry: dict[tuple[int, int], dict[str, Any]] = {}
        self._backend: dict[str, Any] | None = None
        self._fidelity: dict[str, Any] | None = None

    def _capture_stats(self) -> dict[str, Any]:
        snapshot = getattr(self.source, "stats_snapshot", None)
        if not callable(snapshot):
            return {
                "queue_policy": "sequential",
                "captured_frames": self.metrics.snapshot().captured_frames,
                "dropped_frames": 0,
            }
        stats = dict(snapshot())
        stats.setdefault("queue_policy", "drop-oldest")
        return stats

    def _sync_capture_metrics(self) -> None:
        stats = self._capture_stats()
        if "capture_elapsed_ms" in stats:
            self.metrics.sync_capture_stats(stats)

    def _set_paused(self, paused: bool) -> None:
        method = getattr(self.source, "pause" if paused else "resume", None)
        if callable(method):
            method()

    @staticmethod
    def _is_pause_key(key: int) -> bool:
        return key in (ord("p"), ord(" "))

    @staticmethod
    def _is_quit_key(key: int) -> bool:
        return key in (ord("q"), 27)

    def run(self) -> dict[str, Any]:
        started_ns = self.clock_ns()
        iterator = iter(self.source)
        exit_reason = "source-exhausted"
        paused = False
        base_frame: NDArray[np.uint8] | None = None
        detections = ()
        current_source_label = self.source_label
        try:
            self.display.open(self.window_title)
            while True:
                try:
                    frame = next(iterator)
                except StopIteration:
                    exit_reason = "source-exhausted"
                    break
                source_timestamp = (
                    frame.capture_monotonic_ns
                    if frame.capture_monotonic_ns is not None
                    else self.clock_ns()
                )
                self.metrics.note_source(source_timestamp)
                adapted = adapt_frame(frame.image)
                source_size = tuple(adapted.manifest["source_size"])
                self._geometry[source_size] = adapted.manifest
                exact_frame = MediaFrame(
                    image=adapted.image,
                    index=frame.index,
                    source_kind=frame.source_kind,
                    path=frame.path,
                    timestamp_seconds=frame.timestamp_seconds,
                    capture_monotonic_ns=frame.capture_monotonic_ns,
                    decode_elapsed_ns=frame.decode_elapsed_ns,
                )
                inference_started = self.clock_ns()
                result = self.exact_session.infer_live_frame(
                    exact_frame,
                    run_id=f"{self.run_id}-f{frame.index:06d}",
                )
                inference_elapsed = self.clock_ns() - inference_started
                self.metrics.note_inference(max(1, inference_elapsed))
                self._backend = dict(result.backend)
                self._fidelity = dict(result.fidelity)
                base_frame = adapted.image
                detections = result.detections
                current_source_label = self.source_label
                if frame.source_id is not None:
                    current_source_label += f" | camera {frame.source_id}"
                self._sync_capture_metrics()
                if self.writer is not None:
                    self.metrics.note_recorded()
                self.metrics.note_display(self.clock_ns())
                rendered = render_frame(
                    base_frame,
                    detections,
                    self.metrics.snapshot(),
                    source_label=current_source_label,
                    paused=False,
                    recording=self.writer is not None,
                    banner=self.banner,
                )
                if self.writer is not None:
                    self.writer.write(rendered)
                self.display.show(rendered)

                if (
                    self.max_frames is not None
                    and self.metrics.snapshot().inferred_frames >= self.max_frames
                ):
                    exit_reason = "frame-limit"
                    break
                key = self.display.poll_key(1)
                if self._is_quit_key(key):
                    exit_reason = "key-quit"
                    break
                if not self.display.is_open():
                    exit_reason = "window-closed"
                    break
                if self._is_pause_key(key):
                    paused = True
                    self._set_paused(True)
                    while paused:
                        paused_frame = render_frame(
                            base_frame,
                            detections,
                            self.metrics.snapshot(),
                            source_label=current_source_label,
                            paused=True,
                            recording=self.writer is not None,
                            banner=self.banner,
                        )
                        self.display.show(paused_frame)
                        pause_key = self.display.poll_key(25)
                        if self._is_quit_key(pause_key):
                            exit_reason = "key-quit"
                            paused = False
                            break
                        if not self.display.is_open():
                            exit_reason = "window-closed"
                            paused = False
                            break
                        if self._is_pause_key(pause_key):
                            paused = False
                            self._set_paused(False)
                    if exit_reason in {"key-quit", "window-closed"}:
                        break
        except KeyboardInterrupt:
            exit_reason = "keyboard-interrupt"
        finally:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()
            if self.writer is not None:
                self.writer.release()
            self.display.close()

        self._sync_capture_metrics()
        ended_ns = self.clock_ns()
        metrics = asdict(self.metrics.snapshot())
        capture = self._capture_stats()
        summary: dict[str, Any] = {
            "schema": "verkeye.live-viewer-session.v1",
            "run_id": self.run_id,
            "source": self.source_label,
            "banner": self.banner,
            "exit_reason": exit_reason,
            "elapsed_ns": max(0, ended_ns - started_ns),
            "metrics": metrics,
            "capture": capture,
            "geometry": [
                self._geometry[key] for key in sorted(self._geometry)
            ],
            "backend": self._backend,
            "fidelity": self._fidelity,
            "last_detections": list(annotation_records(detections)),
            "recording": (
                {**self.recording, "frames_written": metrics["recorded_frames"]}
                if self.recording is not None
                else None
            ),
            "controls": {
                "pause_resume": ["space", "p"],
                "quit": ["q", "escape", "window-close"],
            },
        }
        if self.summary_path is not None:
            atomic_write_json(self.summary_path, summary)
        return summary
