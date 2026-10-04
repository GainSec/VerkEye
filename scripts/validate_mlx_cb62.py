#!/usr/bin/env python3
"""Validate and benchmark the complete recovered CB62 graph on MLX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Mapping

import numpy as np

from verkeye.accelerated.cb62 import Cb62MlxSession, MlxCb62Runtime
from verkeye.compat.ades import (
    assemble_cb62_predictions,
    decode_float32_chw,
    parse_cavalry_verbose,
)
from verkeye.runtime.inference import load_cb62_core_profile
from verkeye.runtime.preprocess import preprocess_cb62_exact_bgr
from verkeye.runtime.yolov6 import decode_cb62_raw_head_core


REPOSITORY = Path(__file__).resolve().parents[1]
ORACLE_IMAGE = (
    "xiezhouyi/cnngen@"
    "sha256:4c4266444740d9d52f6dd695fff86763ebb51dfb01b0c9778ca20209308cdcdb"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root", type=Path, default=REPOSITORY / ".runtime/ades-full-kernels"
    )
    parser.add_argument(
        "--split4-root", type=Path, default=REPOSITORY / ".runtime/ades-split4"
    )
    parser.add_argument(
        "--split5-root", type=Path, default=REPOSITORY / ".runtime/ades-split5"
    )
    parser.add_argument(
        "--input", type=Path, default=REPOSITORY / ".runtime/ades-direct/d12.bin"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=REPOSITORY / ".runtime/ades-cb62/benchmark-input.png",
    )
    parser.add_argument(
        "--pipeline-evidence",
        type=Path,
        default=REPOSITORY / "evidence/compatibility/cvproc-yolov6-pipeline.json",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-cb62/kernel-dump-run",
    )
    parser.add_argument(
        "--additional-oracle-root",
        type=Path,
        action="append",
        default=[],
        help="validate an additional ADES run directory (repeatable)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY / "evidence/compatibility/cavalry-yolov6n-hor-verbose.txt",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-compatibility-report.json",
    )
    parser.add_argument(
        "--performance-out",
        type=Path,
        default=REPOSITORY / "evidence/performance/macos-accelerated-final.json",
    )
    return parser


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _percentile(samples: list[int], percentile: int) -> int:
    ordered = sorted(samples)
    return ordered[max(0, (percentile * len(ordered) + 99) // 100 - 1)]


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 1 or args.frames < 1:
        raise SystemExit("warmup and frames must be positive")

    session = Cb62MlxSession(
        capture_root=args.capture_root,
        split4_root=args.split4_root,
        split5_root=args.split5_root,
    )
    source = np.fromfile(args.input, dtype=np.uint8).reshape(3, 608, 1088)
    actual = session.infer_outputs(source)
    manifest = parse_cavalry_verbose(args.manifest.read_text(encoding="utf-8"))

    def compare_oracle(
        oracle_root: Path, observed_outputs: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
        expected_outputs: dict[str, np.ndarray] = {}
        compared: list[dict[str, object]] = []
        for split_index in (8, 9):
            for output_index, port in enumerate(manifest.splits[split_index].outputs):
                if not port.name.startswith("output_"):
                    continue
                expected = decode_float32_chw(
                    (
                        oracle_root
                        / f"split-{split_index:02}-output-{output_index:02}.bin"
                    ).read_bytes(),
                    port,
                )
                expected_outputs[port.name] = expected
                observed = observed_outputs[port.name]
                passed = np.array_equal(observed, expected)
                compared.append(
                    {
                        "name": port.name,
                        "split_index": split_index,
                        "output_index": output_index,
                        "shape": list(observed.shape),
                        "dtype": observed.dtype.name,
                        "oracle_sha256": _sha256(expected.tobytes(order="C")),
                        "actual_sha256": _sha256(observed.tobytes(order="C")),
                        "mismatch_count": int(
                            np.count_nonzero(observed != expected)
                        ),
                        "comparison": "byte-identical" if passed else "different",
                        "passed": passed,
                    }
                )
        return expected_outputs, compared

    oracle, boundaries = compare_oracle(args.oracle_root, actual)
    if len(boundaries) != 6 or not all(item["passed"] for item in boundaries):
        raise SystemExit("complete MLX output differs from the ADES oracle")

    actual_predictions = assemble_cb62_predictions(actual)
    oracle_predictions = assemble_cb62_predictions(oracle)
    predictions_passed = np.array_equal(actual_predictions, oracle_predictions)
    if not predictions_passed:
        raise SystemExit("complete MLX prediction matrix differs from ADES")

    oracle_cases = [
        {
            "name": args.oracle_root.name,
            "input_sha256": _sha256(source.tobytes(order="C")),
            "public_output_boundaries": boundaries,
        }
    ]
    for oracle_root in args.additional_oracle_root:
        additional_source = np.fromfile(
            oracle_root / "input.tensor", dtype=np.uint8
        ).reshape(3, 608, 1088)
        additional_actual = session.infer_outputs(additional_source)
        _, additional_boundaries = compare_oracle(
            oracle_root, additional_actual
        )
        if not all(item["passed"] for item in additional_boundaries):
            raise SystemExit(
                f"complete MLX output differs from ADES for {oracle_root.name}"
            )
        oracle_cases.append(
            {
                "name": oracle_root.name,
                "input_sha256": _sha256(additional_source.tobytes(order="C")),
                "public_output_boundaries": additional_boundaries,
            }
        )

    validation = {
        "schema": "verkeye.cb62-accelerated-validation.v1",
        "status": "passed",
        "model_sha256": session.MODEL_SHA256,
        "oracle_image": ORACLE_IMAGE,
        "input": {
            "shape": list(source.shape),
            "dtype": source.dtype.name,
            "sha256": _sha256(source.tobytes(order="C")),
        },
        "backend": dict(session.identity),
        "oracle_cases": oracle_cases,
        "public_output_boundaries": boundaries,
        "prediction_matrix": {
            "shape": list(actual_predictions.shape),
            "dtype": actual_predictions.dtype.name,
            "oracle_sha256": _sha256(oracle_predictions.tobytes(order="C")),
            "actual_sha256": _sha256(actual_predictions.tobytes(order="C")),
            "mismatch_count": int(
                np.count_nonzero(actual_predictions != oracle_predictions)
            ),
            "comparison": "byte-identical",
            "passed": True,
        },
        "split_proof_evidence": [
            "evidence/accelerated/cb62-split"
            f"{index:02}-{'fastconv' if index < 4 else 'mixed'}-validation.json"
            for index in range(10)
        ],
    }
    args.validation_out.parent.mkdir(parents=True, exist_ok=True)
    args.validation_out.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    import cv2

    decoded = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if decoded is None:
        raise SystemExit(f"cannot decode benchmark image {args.image}")
    runtime = MlxCb62Runtime(session=session)
    profile = load_cb62_core_profile(args.pipeline_evidence)

    def infer_frame(index: int):
        prepared = preprocess_cb62_exact_bgr(decoded)
        result = runtime.infer(
            prepared.tensor.tobytes(), run_id=f"benchmark-{index:06d}"
        )
        detections = decode_cb62_raw_head_core(result.predictions, profile)
        return result, detections

    for index in range(args.warmup):
        infer_frame(index)
    samples: list[int] = []
    started = time.perf_counter_ns()
    final = None
    final_detections = None
    for index in range(args.frames):
        frame_started = time.perf_counter_ns()
        final, final_detections = infer_frame(index + args.warmup)
        samples.append(time.perf_counter_ns() - frame_started)
    elapsed_ns = time.perf_counter_ns() - started
    assert final is not None
    assert final_detections is not None
    if any(
        not np.array_equal(final.tensors[name], expected)
        for name, expected in oracle.items()
    ):
        raise SystemExit("warmed MLX output differs from the ADES oracle")
    performance = {
        "schema": "verkeye.performance-sample.v1",
        "status": "passed" if args.frames * 1_000_000_000 / elapsed_ns >= 30 else "failed",
        "target_fps": 30,
        "scope": (
            "exact preprocess, device-resident recovered model, prediction "
            "assembly, output hashing, and binary-proved detector postprocess"
        ),
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "elapsed_ns": elapsed_ns,
        "fps": args.frames * 1_000_000_000 / elapsed_ns,
        "latency_ns": {
            "minimum": min(samples),
            "p50": _percentile(samples, 50),
            "p95": _percentile(samples, 95),
            "maximum": max(samples),
            "mean": statistics.fmean(samples),
        },
        "raw_samples_ns": samples,
        "backend": dict(session.identity),
        "benchmark_image": {
            "path": args.image.name,
            "sha256": _sha256(args.image.read_bytes()),
            "shape": list(decoded.shape),
        },
        "final_detection_count": len(final_detections),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "output_compatibility": "byte-identical after warmup and measurement",
    }
    args.performance_out.parent.mkdir(parents=True, exist_ok=True)
    args.performance_out.write_text(
        json.dumps(performance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"validation": validation["status"], "fps": performance["fps"], "performance": performance["status"]}, sort_keys=True))
    return 0 if performance["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
