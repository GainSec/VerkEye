"""Fixed one-command GUI demonstrations for the exact recovered CB62 model."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sysconfig
import time
from typing import Any
from uuid import uuid4

import numpy as np

from .accelerated.cb62 import (
    AcceleratedCb62Runtime,
    Cb62AcceleratedError,
    Cb62MlxSession,
)
from .accelerated.openvino import OpenVinoCb62Session
from .compat.ades_runtime import load_runtime_spec
from .runtime.demo_sources import (
    ProductionDemoFixture,
    SingaporeTrafficStream,
    resolve_production_demo_fixture,
)
from .runtime.inference import ExactFrameSession
from .runtime.generation import generated_runtime_path, load_generated_runtime
from .runtime.sources import MediaFrame, MediaSourceError
from .viewer.controller import LiveViewerSession, OpenCvDisplay


PRODUCTION_DEMO_EXPECTED_CLASS_ID = 0
PRODUCTION_DEMO_MINIMUM_CONFIDENCE = 0.1005


@dataclass(frozen=True, slots=True)
class DemoRuntimePaths:
    model: Path
    runtime_spec: Path
    pipeline_evidence: Path
    generated_runtime_root: Path


@dataclass(frozen=True, slots=True)
class DemoPreset:
    number: int
    profile: str
    source_label: str
    banner: str
    window_title: str

    def make_source(self) -> SingaporeTrafficStream | "LoopingFixtureSource":
        if self.number == 1:
            return SingaporeTrafficStream()
        if self.number == 2:
            return LoopingFixtureSource(resolve_production_demo_fixture())
        raise ValueError(f"unsupported demo: {self.number}")


def demo_preset(number: int) -> DemoPreset:
    if number == 1:
        return DemoPreset(
            number=1,
            profile="small-object",
            source_label="Singapore LTA traffic stills",
            banner="DEMO 1 | SINGAPORE LTA | SMALL-OBJECT PROFILE",
            window_title="VerkEye Demo 1 — Singapore traffic / small-object",
        )
    if number == 2:
        return DemoPreset(
            number=2,
            profile="production",
            source_label="Bundled CC0 street scene",
            banner="DEMO 2 | BUNDLED OFFLINE SCENE | PRODUCTION PROFILE",
            window_title="VerkEye Demo 2 — bundled scene / production",
        )
    raise ValueError(f"unsupported demo: {number}")


def _first_existing(candidates: tuple[Path, ...], *, label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise MediaSourceError(
        f"Demo runtime {label} is unavailable; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_demo_runtime_paths() -> DemoRuntimePaths:
    repository = Path(__file__).resolve().parents[2]
    shared = Path(sysconfig.get_path("data")) / "share" / "verkeye"
    model = _first_existing(
        tuple(
            (
                [Path(os.environ["VERKEYE_MODEL"])]
                if os.environ.get("VERKEYE_MODEL")
                else []
            )
            + [
                repository / "fixtures/models/yolov6n_hor.bin",
                shared / "models/yolov6n_hor.bin",
            ]
        ),
        label="model",
    )
    runtime_spec = _first_existing(
        (
            repository / "config/cb62-ades-runtime.json",
            shared / "ades/cb62-ades-runtime.json",
        ),
        label="runtime specification",
    )
    pipeline = _first_existing(
        (
            repository / "evidence/compatibility/cvproc-yolov6-pipeline.json",
            shared / "ades/cvproc-yolov6-pipeline.json",
        ),
        label="pipeline evidence",
    )
    spec = load_runtime_spec(runtime_spec)
    generated_base = Path(
        os.environ.get("VERKEYE_RUNTIME_BASE", repository / ".runtime/generated")
    )
    return DemoRuntimePaths(
        model=model,
        runtime_spec=runtime_spec,
        pipeline_evidence=pipeline,
        generated_runtime_root=_first_existing(
            (generated_runtime_path(generated_base, spec.model_sha256),),
            label="owner-generated accelerated runtime",
        ),
    )


class LoopingFixtureSource(Iterator[MediaFrame]):
    """Decode the pinned fixture once and yield it until the viewer closes."""

    def __init__(self, fixture: ProductionDemoFixture) -> None:
        import cv2

        image = cv2.imread(str(fixture.path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise MediaSourceError("production demo fixture could not be decoded")
        self.fixture = fixture
        self.image: np.ndarray[Any, np.dtype[np.uint8]] = image
        self.index = 0
        self.closed = False

    def __iter__(self) -> "LoopingFixtureSource":
        return self

    def __next__(self) -> MediaFrame:
        if self.closed:
            raise StopIteration
        frame = MediaFrame(
            image=self.image,
            index=self.index,
            source_kind="image",
            path=self.fixture.path,
            timestamp_seconds=None,
            capture_monotonic_ns=time.monotonic_ns(),
            source_id="demo-2-production-scene",
            source_url=str(self.fixture.metadata["source_page"]),
            source_timestamp=None,
        )
        self.index += 1
        return frame

    def close(self) -> None:
        self.closed = True

    def stats_snapshot(self) -> dict[str, object]:
        return {
            "source": "bundled production demo fixture",
            "queue_policy": "looping-single-frame",
            "captured_frames": self.index,
            "dropped_frames": 0,
            "license": self.fixture.metadata["license"],
            "source_page": self.fixture.metadata["source_page"],
            "sha256": self.fixture.metadata["sha256"],
            "closed": self.closed,
        }


def _build_exact_session(
    preset: DemoPreset,
    paths: DemoRuntimePaths,
) -> ExactFrameSession:
    spec = load_runtime_spec(paths.runtime_spec)
    generated_runtime = load_generated_runtime(
        paths.generated_runtime_root,
        expected_model_sha256=spec.model_sha256,
    )
    if importlib.util.find_spec("mlx") is not None:
        provider: Any = Cb62MlxSession(generated_runtime=generated_runtime)
    elif importlib.util.find_spec("openvino") is not None:
        import openvino as ov

        devices = tuple(ov.Core().available_devices)
        device = "GPU" if "GPU" in devices else "CPU"
        provider = OpenVinoCb62Session(
            generated_runtime=generated_runtime,
            device=device,
        )
    else:
        raise Cb62AcceleratedError(
            "Demo requires the MLX (macOS) or OpenVINO (Linux) runtime extra"
        )
    return ExactFrameSession(
        model=paths.model,
        backend=AcceleratedCb62Runtime(session=provider),
        runtime_spec=spec,
        pipeline_evidence=paths.pipeline_evidence,
        operating_profile=preset.profile,
    )


def run_demo(
    number: int,
    *,
    summary_path: str | Path | None = None,
    max_frames: int | None = None,
    source_factory: Callable[[DemoPreset], object] | None = None,
    session_factory: Callable[[DemoPreset, DemoRuntimePaths], object] | None = None,
    display_factory: Callable[[], object] = OpenCvDisplay,
    viewer_factory: Callable[..., object] = LiveViewerSession,
) -> int:
    preset = demo_preset(number)
    paths = resolve_demo_runtime_paths()
    source = (
        source_factory(preset) if source_factory is not None else preset.make_source()
    )
    exact_session = (
        session_factory(preset, paths)
        if session_factory is not None
        else _build_exact_session(preset, paths)
    )
    target = Path(summary_path) if summary_path is not None else Path(
        f"verkeye-demo-{number}-summary.json"
    )
    viewer = viewer_factory(
        source=source,
        exact_session=exact_session,
        display=display_factory(),
        source_label=preset.source_label,
        banner=preset.banner,
        run_id=f"demo-{number}-{uuid4().hex[:12]}",
        summary_path=target,
        max_frames=max_frames,
        window_title=preset.window_title,
    )
    viewer.run()
    print(f"wrote {target}")
    return 0
