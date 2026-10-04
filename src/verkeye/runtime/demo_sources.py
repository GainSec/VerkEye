"""Network and pinned-media sources for the fixed VerkEye demonstrations."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Protocol
from urllib.request import Request, urlopen

import numpy as np

from .sources import MediaFrame, MediaSourceError


SINGAPORE_TRAFFIC_IMAGES_URL = (
    "https://api.data.gov.sg/v1/transport/traffic-images"
)
_DEFAULT_MAX_API_BYTES = 2 * 1024 * 1024
_DEFAULT_MAX_IMAGE_BYTES = 12 * 1024 * 1024
PRODUCTION_DEMO_SHA256 = (
    "459f3696a29606084bbb18e3e2acb2f435f72c31f74f5637c480db0267109741"
)
PRODUCTION_DEMO_SIZE = 365_478


@dataclass(frozen=True, slots=True)
class ProductionDemoFixture:
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def resolve_production_demo_fixture(
    *,
    asset_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> ProductionDemoFixture:
    """Resolve and hash-verify the redistributable Demo 2 scene."""

    asset_root = Path(__file__).resolve().parents[1] / "assets" / "demo"
    asset = Path(asset_path) if asset_path is not None else asset_root / "production-scene.jpg"
    sidecar = (
        Path(metadata_path)
        if metadata_path is not None
        else asset_root / "production-scene.json"
    )
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        payload = asset.read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise MediaSourceError("production demo fixture is unavailable") from error
    if not isinstance(metadata, dict):
        raise MediaSourceError("production demo metadata is not an object")
    if len(payload) != PRODUCTION_DEMO_SIZE or metadata.get("bytes") != len(payload):
        raise MediaSourceError("production demo fixture size does not match metadata")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != PRODUCTION_DEMO_SHA256 or metadata.get("sha256") != digest:
        raise MediaSourceError("production demo fixture digest does not match metadata")
    return ProductionDemoFixture(
        path=asset.resolve(),
        metadata_path=sidecar.resolve(),
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class HttpPayload:
    body: bytes
    content_type: str


class HttpTransport(Protocol):
    def get(self, url: str, *, timeout: float, max_bytes: int) -> HttpPayload: ...


class UrllibTransport:
    """Small bounded HTTP transport with no third-party network dependency."""

    def get(self, url: str, *, timeout: float, max_bytes: int) -> HttpPayload:
        request = Request(
            url,
            headers={
                "Accept": "application/json,image/jpeg",
                "User-Agent": "VerkEye/0.1 demo",
            },
        )
        with urlopen(request, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise MediaSourceError(
                    f"Singapore response exceeds {max_bytes} bytes: {url}"
                )
            content_type = response.headers.get("Content-Type", "")
        return HttpPayload(body=body, content_type=content_type)


@dataclass(frozen=True, slots=True)
class SingaporeCamera:
    camera_id: str
    image_url: str
    timestamp: str
    snapshot_timestamp: str


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise MediaSourceError(f"Singapore camera has no valid {key}")
    return str(value).strip()


def parse_singapore_snapshot(document: object) -> tuple[SingaporeCamera, ...]:
    """Validate the official data.gov.sg response and return stable ordering."""

    if not isinstance(document, dict):
        raise MediaSourceError("Singapore response is not a JSON object")
    items = document.get("items")
    if not isinstance(items, list) or not items:
        raise MediaSourceError("Singapore response contains no snapshots")
    snapshot = items[0]
    if not isinstance(snapshot, dict):
        raise MediaSourceError("Singapore snapshot is not an object")
    snapshot_timestamp = _required_text(snapshot, "timestamp")
    records = snapshot.get("cameras")
    if not isinstance(records, list) or not records:
        raise MediaSourceError("Singapore snapshot contains no cameras")

    cameras: list[SingaporeCamera] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise MediaSourceError("Singapore camera record is not an object")
        camera_id = _required_text(record, "camera_id")
        image_url = _required_text(record, "image")
        if not image_url.startswith("https://"):
            raise MediaSourceError(
                f"Singapore camera {camera_id} image URL is not HTTPS"
            )
        if camera_id in seen:
            raise MediaSourceError(
                f"Singapore response repeats camera ID {camera_id}"
            )
        seen.add(camera_id)
        cameras.append(
            SingaporeCamera(
                camera_id=camera_id,
                image_url=image_url,
                timestamp=_required_text(record, "timestamp"),
                snapshot_timestamp=snapshot_timestamp,
            )
        )
    return tuple(
        sorted(
            cameras,
            key=lambda camera: (
                0,
                int(camera.camera_id),
            )
            if camera.camera_id.isdigit()
            else (1, camera.camera_id),
        )
    )


class SingaporeTrafficStream(Iterator[MediaFrame]):
    """Repeatedly poll and decode current Singapore LTA traffic stills."""

    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        api_url: str = SINGAPORE_TRAFFIC_IMAGES_URL,
        timeout: float = 15.0,
        camera_hold_seconds: float = 2.0,
        refresh_seconds: float = 60.0,
        retry_delays: tuple[float, ...] = (1.0, 2.0, 4.0),
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if camera_hold_seconds < 0 or refresh_seconds < 0:
            raise ValueError("demo timing values cannot be negative")
        if any(delay < 0 for delay in retry_delays):
            raise ValueError("retry delays cannot be negative")
        self.transport = transport or UrllibTransport()
        self.api_url = api_url
        self.timeout = timeout
        self.camera_hold_seconds = camera_hold_seconds
        self.refresh_seconds = refresh_seconds
        self.retry_delays = retry_delays
        self.sleep = sleep
        self.monotonic = monotonic
        self._pending: deque[SingaporeCamera] = deque()
        self._cameras: tuple[SingaporeCamera, ...] = ()
        self._next_poll_at = 0.0
        self._closed = False
        self._index = 0
        self._first_frame = True
        self._completed_polls = 0
        self._retry_count = 0
        self._decoded_frames = 0
        self._current_camera: SingaporeCamera | None = None

    def __iter__(self) -> "SingaporeTrafficStream":
        return self

    def _get_with_retries(self, url: str, *, max_bytes: int) -> HttpPayload:
        attempts = len(self.retry_delays) + 1
        for attempt in range(attempts):
            if self._closed:
                raise StopIteration
            try:
                return self.transport.get(
                    url,
                    timeout=self.timeout,
                    max_bytes=max_bytes,
                )
            except (OSError, TimeoutError) as error:
                if attempt == attempts - 1:
                    raise MediaSourceError(
                        f"Singapore request failed after {attempts} attempts: {url}"
                    ) from error
                delay = self.retry_delays[attempt]
                self._retry_count += 1
                self.sleep(delay)
        raise AssertionError("unreachable retry loop")

    def _poll(self) -> None:
        response = self._get_with_retries(
            self.api_url,
            max_bytes=_DEFAULT_MAX_API_BYTES,
        )
        if "json" not in response.content_type.casefold():
            raise MediaSourceError(
                "Singapore API did not return application/json"
            )
        try:
            document = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MediaSourceError("Singapore API returned malformed JSON") from error
        self._cameras = parse_singapore_snapshot(document)
        self._pending.extend(self._cameras)
        self._completed_polls += 1
        self._next_poll_at = self.monotonic() + self.refresh_seconds

    def _decode(self, camera: SingaporeCamera) -> MediaFrame:
        response = self._get_with_retries(
            camera.image_url,
            max_bytes=_DEFAULT_MAX_IMAGE_BYTES,
        )
        media_type = response.content_type.partition(";")[0].strip().casefold()
        allowed_media_types = {
            "image/jpeg",
            "image/jpg",
            "application/octet-stream",
        }
        if (
            media_type not in allowed_media_types
            or not response.body.startswith(b"\xff\xd8\xff")
        ):
            raise MediaSourceError(
                f"Singapore camera {camera.camera_id} did not return JPEG data"
            )
        import cv2

        started = time.monotonic_ns()
        encoded = np.frombuffer(response.body, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        elapsed = time.monotonic_ns() - started
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise MediaSourceError(
                f"Singapore camera {camera.camera_id} JPEG could not be decoded"
            )
        frame = MediaFrame(
            image=image,
            index=self._index,
            source_kind="image",
            path=None,
            timestamp_seconds=None,
            capture_monotonic_ns=time.monotonic_ns(),
            decode_elapsed_ns=elapsed,
            source_id=camera.camera_id,
            source_url=camera.image_url,
            source_timestamp=camera.timestamp,
        )
        self._index += 1
        self._decoded_frames += 1
        self._current_camera = camera
        return frame

    def __next__(self) -> MediaFrame:
        if self._closed:
            raise StopIteration
        if not self._pending:
            if not self._cameras or self.monotonic() >= self._next_poll_at:
                self._poll()
            else:
                self._pending.extend(self._cameras)
        if not self._first_frame:
            self.sleep(self.camera_hold_seconds)
        self._first_frame = False
        return self._decode(self._pending.popleft())

    def close(self) -> None:
        self._closed = True
        self._pending.clear()
        self._cameras = ()

    @property
    def current_camera(self) -> SingaporeCamera | None:
        return self._current_camera

    def stats_snapshot(self) -> dict[str, object]:
        return {
            "source": "Singapore LTA traffic images",
            "api_url": self.api_url,
            "completed_polls": self._completed_polls,
            "decoded_frames": self._decoded_frames,
            "retry_count": self._retry_count,
            "current_camera_id": (
                self._current_camera.camera_id if self._current_camera else None
            ),
            "refresh_seconds": self.refresh_seconds,
            "closed": self._closed,
        }
