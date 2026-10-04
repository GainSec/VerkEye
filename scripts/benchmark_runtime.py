#!/usr/bin/env python3
"""Benchmark complete warmed VerkEye inference without hiding frame work."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np

from verkeye.compat.ades_runtime import (
    AdesRuntimeError,
    DockerAdesRuntime,
    load_runtime_spec,
)
from verkeye.evidence import atomic_write_json
from verkeye.runtime.inference import run_exact_image
from verkeye.runtime.performance import sample_from_frame_documents


REPOSITORY = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ades",), default="ades")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=REPOSITORY / "fixtures/models/yolov6n_hor.bin",
    )
    parser.add_argument(
        "--runtime-spec",
        type=Path,
        default=REPOSITORY / "config/cb62-ades-runtime.json",
    )
    parser.add_argument(
        "--pipeline-evidence",
        type=Path,
        default=REPOSITORY
        / "evidence/compatibility/cvproc-yolov6-pipeline.json",
    )
    parser.add_argument(
        "--executor-source",
        type=Path,
        default=REPOSITORY / "src/verkeye/compat/native/ades_executor.cpp",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=REPOSITORY / ".runtime/ades-cb62",
    )
    parser.add_argument("--prepare", action="store_true")
    return parser


def _benchmark_image(path: Path) -> str:
    import cv2

    y, x = np.indices((608, 1088), dtype=np.uint16)
    image = np.stack(
        ((x % 256), (y % 256), ((x + y) % 256)),
        axis=2,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write benchmark image {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    spec = load_runtime_spec(args.runtime_spec)
    backend = DockerAdesRuntime(
        spec=spec,
        model=args.model,
        workspace=args.workspace,
        executor_source=args.executor_source,
    )
    if args.prepare:
        backend.prepare()
    else:
        try:
            backend.prepared()
        except AdesRuntimeError:
            backend.prepare()

    image_path = args.workspace / "benchmark-input.png"
    image_sha256 = _benchmark_image(image_path)
    batch_id = uuid4().hex[:12]
    for index in range(args.warmup):
        run_exact_image(
            model=args.model,
            image=image_path,
            backend=backend,
            runtime_spec=spec,
            pipeline_evidence=args.pipeline_evidence,
            run_id=f"bench-{batch_id}-warmup-{index:04d}",
        )

    measured: list[dict[str, object]] = []
    started = time.monotonic_ns()
    for index in range(args.frames):
        measured.append(
            run_exact_image(
                model=args.model,
                image=image_path,
                backend=backend,
                runtime_spec=spec,
                pipeline_evidence=args.pipeline_evidence,
                run_id=f"bench-{batch_id}-measured-{index:04d}",
            )
        )
    elapsed_ns = time.monotonic_ns() - started
    sample = sample_from_frame_documents(
        measured,
        warmup_frames=args.warmup,
        elapsed_ns=elapsed_ns,
        backend_identity={
            "name": "ambarella-ades-host-emulator",
            "image": spec.image,
            "cavalry_version": spec.cavalry_version,
            "cavalry_hash": spec.cavalry_hash,
        },
        host_identity={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    )
    document = {
        "schema": "verkeye.runtime-performance.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": spec.model_sha256,
        "pipeline_evidence_sha256": spec.pipeline_evidence_sha256,
        "input": {
            "width": 1088,
            "height": 608,
            "decoded_format": "BGR uint8",
            "png_sha256": image_sha256,
        },
        "performance": sample.to_document(),
        "frame_run_ids": [str(item["run_id"]) for item in measured],
    }
    atomic_write_json(args.json_out, document)
    print(f"wrote {args.json_out}")
    print(f"effective_fps={sample.fps:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
