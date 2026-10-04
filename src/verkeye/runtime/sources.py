"""Deterministic local image, frame-directory, video, and webcam sources."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray


SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


class MediaSourceError(ValueError):
    """A local media source cannot be opened or decoded."""


class Capture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...

    def get(self, property_id: int) -> float: ...

    def release(self) -> None: ...


ImageDecoder = Callable[[str], NDArray[np.uint8] | None]
CaptureFactory = Callable[[Any], Capture]


@dataclass(frozen=True, slots=True)
class MediaFrame:
    image: NDArray[np.uint8]
    index: int
    source_kind: Literal["image", "frame-directory", "video", "webcam"]
    path: Path | None
    timestamp_seconds: float | None
    capture_monotonic_ns: int | None = None
    decode_elapsed_ns: int | None = None
    source_id: str | None = None
    source_url: str | None = None
    source_timestamp: str | None = None


def _default_decoder(path: str) -> NDArray[np.uint8] | None:
    import cv2

    return cv2.imread(path, cv2.IMREAD_COLOR)


def _default_capture_factory(source: object) -> Capture:
    import cv2

    return cv2.VideoCapture(source)


def _natural_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def _decode(path: Path, decoder: ImageDecoder) -> NDArray[np.uint8]:
    image = decoder(str(path))
    if image is None:
        raise MediaSourceError(f"could not decode image: {path}")
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise MediaSourceError(f"decoded image is not HxWx3: {path}")
    return image


def read_image(
    path: str | Path,
    *,
    decoder: ImageDecoder | None = None,
) -> MediaFrame:
    source = Path(path)
    if not source.is_file():
        raise MediaSourceError(f"image does not exist: {source}")
    started = time.monotonic_ns()
    image = _decode(source, decoder or _default_decoder)
    elapsed_ns = time.monotonic_ns() - started
    return MediaFrame(
        image,
        0,
        "image",
        source.resolve(),
        None,
        decode_elapsed_ns=elapsed_ns,
    )


def iter_frame_directory(
    path: str | Path,
    *,
    decoder: ImageDecoder | None = None,
) -> Iterator[MediaFrame]:
    source = Path(path)
    if not source.is_dir():
        raise MediaSourceError(f"frame directory does not exist: {source}")
    active_decoder = decoder or _default_decoder
    frames = sorted(
        (
            item
            for item in source.iterdir()
            if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=_natural_key,
    )
    if not frames:
        raise MediaSourceError(
            f"frame directory contains no supported images: {source}"
        )
    for index, frame_path in enumerate(frames):
        started = time.monotonic_ns()
        image = _decode(frame_path, active_decoder)
        elapsed_ns = time.monotonic_ns() - started
        yield MediaFrame(
            image,
            index,
            "frame-directory",
            frame_path.resolve(),
            None,
            decode_elapsed_ns=elapsed_ns,
        )


def iter_capture(
    source: str | Path | int,
    *,
    capture_factory: CaptureFactory | None = None,
    max_frames: int | None = None,
) -> Iterator[MediaFrame]:
    """Iterate a video file or webcam and always release the capture handle."""

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    factory = capture_factory or _default_capture_factory
    capture_source: object = str(source) if isinstance(source, Path) else source
    capture = factory(capture_source)
    source_kind: Literal["video", "webcam"] = (
        "webcam" if isinstance(source, int) else "video"
    )
    source_path = (
        None if source_kind == "webcam" else Path(str(source)).expanduser().resolve()
    )
    try:
        if not capture.isOpened():
            raise MediaSourceError(f"could not open {source_kind} source: {source}")
        index = 0
        while max_frames is None or index < max_frames:
            started = time.monotonic_ns()
            ok, image = capture.read()
            elapsed_ns = time.monotonic_ns() - started
            if not ok:
                break
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                raise MediaSourceError(
                    f"{source_kind} frame {index} is not a decoded HxWx3 image"
                )
            timestamp_msec = float(capture.get(0))
            yield MediaFrame(
                image,
                index,
                source_kind,
                source_path,
                timestamp_msec / 1000.0 if timestamp_msec >= 0 else None,
                decode_elapsed_ns=elapsed_ns,
            )
            index += 1
    finally:
        capture.release()
