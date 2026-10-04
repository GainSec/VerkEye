"""Bounded live capture with explicit backpressure and drop accounting."""

from __future__ import annotations

from collections import deque
from threading import Condition, Event, Lock, Thread, current_thread
import time
from typing import Any

from .sources import Capture, CaptureFactory, MediaFrame, MediaSourceError


class LiveCaptureStream:
    """Read a webcam concurrently and retain only the newest bounded frames.

    The exact CB62 backend is much slower than ordinary camera capture. A
    drop-oldest queue prevents unbounded latency and memory growth while
    retaining source frame indices that account for every dropped frame.
    """

    def __init__(
        self,
        source: int,
        *,
        queue_capacity: int,
        max_frames: int | None,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.source = source
        self.queue_capacity = queue_capacity
        self.max_frames = max_frames
        self._capture_factory = capture_factory
        self._capture: Capture | None = None
        self._queue: deque[MediaFrame] = deque()
        self._condition = Condition()
        self._stop = Event()
        self._resume = Event()
        self._resume.set()
        self._thread: Thread | None = None
        self._release_lock = Lock()
        self._released = False
        self._started_ns: int | None = None
        self._ended_ns: int | None = None
        self._capture_finished = False
        self._error: BaseException | None = None
        self._captured = 0
        self._yielded = 0
        self._dropped = 0
        self._discarded = 0
        self._high_watermark = 0

    def start(self) -> "LiveCaptureStream":
        if self._thread is not None:
            return self
        factory = self._capture_factory
        if factory is None:
            from .sources import _default_capture_factory

            factory = _default_capture_factory
        capture = factory(self.source)
        self._capture = capture
        if not capture.isOpened():
            self._release_capture()
            raise MediaSourceError(f"could not open webcam source: {self.source}")
        self._started_ns = time.monotonic_ns()
        self._thread = Thread(
            target=self._capture_loop,
            name=f"verkeye-webcam-{self.source}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _release_capture(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
            if self._capture is not None:
                self._capture.release()

    def _capture_loop(self) -> None:
        assert self._capture is not None
        try:
            while not self._stop.is_set():
                self._resume.wait()
                if self._stop.is_set():
                    break
                ok, image = self._capture.read()
                if not ok:
                    break
                if image is None or image.ndim != 3 or image.shape[2] != 3:
                    raise MediaSourceError(
                        f"webcam frame {self._captured} is not a decoded HxWx3 image"
                    )
                captured_ns = time.monotonic_ns()
                timestamp_msec = float(self._capture.get(0))
                frame = MediaFrame(
                    image=image,
                    index=self._captured,
                    source_kind="webcam",
                    path=None,
                    timestamp_seconds=(
                        timestamp_msec / 1000.0 if timestamp_msec >= 0 else None
                    ),
                    capture_monotonic_ns=captured_ns,
                )
                with self._condition:
                    self._captured += 1
                    if len(self._queue) == self.queue_capacity:
                        self._queue.popleft()
                        self._dropped += 1
                    self._queue.append(frame)
                    self._high_watermark = max(
                        self._high_watermark, len(self._queue)
                    )
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
        finally:
            self._release_capture()
            with self._condition:
                self._capture_finished = True
                self._ended_ns = time.monotonic_ns()
                self._condition.notify_all()

    def __iter__(self) -> "LiveCaptureStream":
        self.start()
        return self

    def __next__(self) -> MediaFrame:
        self.start()
        if self.max_frames is not None and self._yielded >= self.max_frames:
            self.close()
            raise StopIteration
        with self._condition:
            while not self._queue and not self._capture_finished:
                self._condition.wait()
            if self._queue:
                frame = self._queue.popleft()
                self._yielded += 1
                return frame
            if self._error is not None:
                raise MediaSourceError(
                    f"webcam capture failed: {self._error}"
                ) from self._error
            raise StopIteration

    def close(self) -> None:
        self._stop.set()
        self._resume.set()
        self._release_capture()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=2.0)
        with self._condition:
            self._discarded += len(self._queue)
            self._queue.clear()
            if self._ended_ns is None:
                self._ended_ns = time.monotonic_ns()
            self._condition.notify_all()

    def pause(self) -> None:
        """Pause acquisition before the next capture read."""

        self._resume.clear()

    def resume(self) -> None:
        """Resume acquisition after a viewer pause."""

        self._resume.set()

    def stats_snapshot(self) -> dict[str, Any]:
        with self._condition:
            started = self._started_ns
            ended = self._ended_ns or (time.monotonic_ns() if started else None)
            elapsed_ms = (
                (ended - started) / 1_000_000.0
                if started is not None and ended is not None
                else 0.0
            )
            return {
                "queue_policy": "drop-oldest",
                "queue_capacity": self.queue_capacity,
                "queue_high_watermark": self._high_watermark,
                "captured_frames": self._captured,
                "yielded_frames": self._yielded,
                "dropped_frames": self._dropped,
                "discarded_on_close_frames": self._discarded,
                "queued_frames": len(self._queue),
                "capture_finished": self._capture_finished,
                "paused": not self._resume.is_set(),
                "capture_elapsed_ms": elapsed_ms,
            }

    def __enter__(self) -> "LiveCaptureStream":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
