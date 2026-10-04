#!/usr/bin/env python3
"""Benchmark byte-exact accelerated CB62 execution after an oracle parity gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

from verkeye.accelerated.cb62 import Cb62MlxSession
from verkeye.accelerated.openvino import OpenVinoCb62Session
from verkeye.compat.ades import assemble_cb62_predictions
from verkeye.compat.oracle import load_oracle_bundle
from verkeye.compat.terminal_catalog import validate_terminal_parity_catalog
from verkeye.evidence import atomic_write_json
from verkeye.runtime.inference import load_cb62_core_profile
from verkeye.runtime.preprocess import preprocess_cb62_exact_bgr
from verkeye.runtime.yolov6 import decode_cb62_raw_head_core


REPOSITORY = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mlx", "openvino"), required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--case-id", default="sg-camera-2704")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--openvino-device", default="CPU")
    parser.add_argument("--inference-threads", type=int, default=6)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPOSITORY / "fixtures/oracle/terminal-parity-catalog.json",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-full-kernels",
    )
    parser.add_argument(
        "--split4-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-split4",
    )
    parser.add_argument(
        "--split5-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-split5",
    )
    return parser


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _percentile(samples: list[int], percentile: int) -> int:
    ordered = sorted(samples)
    return ordered[max(0, (percentile * len(ordered) + 99) // 100 - 1)]


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _session(args: argparse.Namespace) -> object:
    common = {
        "capture_root": args.capture_root,
        "split4_root": args.split4_root,
        "split5_root": args.split5_root,
    }
    if args.backend == "mlx":
        return Cb62MlxSession(**common)
    return OpenVinoCb62Session(
        **common,
        device=args.openvino_device,
        inference_threads=args.inference_threads,
    )


def _one_frame(
    session: object,
    image: np.ndarray,
    profile: object,
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[object, ...]]:
    prepared = preprocess_cb62_exact_bgr(image)
    outputs = dict(session.infer_outputs(prepared.tensor[0]))
    predictions = assemble_cb62_predictions(outputs)
    detections = decode_cb62_raw_head_core(predictions, profile)
    return outputs, predictions, detections


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 0 or args.frames <= 0:
        raise ValueError("warmup must be non-negative and frames must be positive")
    if args.target_fps <= 0 or args.inference_threads <= 0:
        raise ValueError("target FPS and inference threads must be positive")

    catalog = json.loads(args.catalog.read_bytes())
    cases = validate_terminal_parity_catalog(catalog)
    case = next((item for item in cases if item["case_id"] == args.case_id), None)
    if case is None:
        raise ValueError(f"unknown registered case: {args.case_id}")
    bundle_path = Path(case["bundle"])
    if not bundle_path.is_absolute():
        bundle_path = REPOSITORY / bundle_path
    bundle = load_oracle_bundle(bundle_path)
    model_input = np.frombuffer(bundle.input_tensor, dtype=np.uint8).reshape(
        3, 608, 1088
    )
    image = np.ascontiguousarray(model_input.transpose(1, 2, 0))
    pipeline_path = Path(catalog["pipeline"]["path"])
    if not pipeline_path.is_absolute():
        pipeline_path = REPOSITORY / pipeline_path
    profile = load_cb62_core_profile(pipeline_path)
    session = _session(args)

    def require_exact(
        outputs: dict[str, np.ndarray],
        predictions: np.ndarray,
        detections: tuple[object, ...],
    ) -> None:
        for name, expected in case["terminal_tensors"].items():
            observed = _sha256(outputs[name].tobytes(order="C"))
            if observed != expected["sha256"]:
                raise RuntimeError(f"{args.case_id}/{name}: terminal parity failed")
        if _sha256(predictions.tobytes(order="C")) != case["prediction_sha256"]:
            raise RuntimeError(f"{args.case_id}: prediction parity failed")
        if list(detections) != case["production_detections"]:
            raise RuntimeError(f"{args.case_id}: detection parity failed")

    require_exact(*_one_frame(session, image, profile))
    for _ in range(args.warmup):
        _one_frame(session, image, profile)

    samples: list[int] = []
    final = None
    started_ns = time.perf_counter_ns()
    for _ in range(args.frames):
        frame_started_ns = time.perf_counter_ns()
        final = _one_frame(session, image, profile)
        samples.append(time.perf_counter_ns() - frame_started_ns)
    elapsed_ns = time.perf_counter_ns() - started_ns
    assert final is not None
    require_exact(*final)

    fps = args.frames * 1_000_000_000 / elapsed_ns
    document = {
        "schema": "verkeye.byte-exact-performance.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if fps >= args.target_fps else "below_target",
        "source_commit": _git_commit(),
        "command": [sys.executable, *sys.argv],
        "correctness_gate": {
            "before_benchmark": "passed",
            "after_benchmark": "passed",
            "terminal_tensor_differing_bytes": 0,
            "prediction_mismatches": 0,
            "production_detection_mismatches": 0,
        },
        "scope": (
            "exact BGR-to-CHW preprocessing, exact recovered model execution, "
            "prediction assembly, and production post-processing"
        ),
        "equivalence_boundary": (
            "identical uint8 CHW bytes entering the recovered CB62 model"
        ),
        "backend": dict(session.identity),
        "model_sha256": catalog["model"]["sha256"],
        "pipeline_sha256": catalog["pipeline"]["sha256"],
        "catalog_sha256": _sha256(args.catalog.read_bytes()),
        "case_id": args.case_id,
        "input_sha256": case["input_sha256"],
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "target_fps": args.target_fps,
        "elapsed_ns": elapsed_ns,
        "fps": fps,
        "latency_ns": {
            "minimum": min(samples),
            "p50": _percentile(samples, 50),
            "p95": _percentile(samples, 95),
            "maximum": max(samples),
            "mean": statistics.fmean(samples),
        },
        "raw_samples_ns": samples,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
    }
    atomic_write_json(args.json_out, document)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "fps": fps,
                "p50_ms": document["latency_ns"]["p50"] / 1_000_000,
                "p95_ms": document["latency_ns"]["p95"] / 1_000_000,
                "status": document["status"],
                "json_out": str(args.json_out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
