#!/usr/bin/env python3
"""Prove and benchmark the recovered CB62 split-9 detector graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from verkeye.accelerated.macos import MlxFastconvSession
from verkeye.compat.ades import decode_float32_chw, parse_cavalry_verbose
from verkeye.cv22.operators import (
    FastconvGeometry,
    channel_halves,
    cv22_sigmoid_float32,
    quantized_silu_int8,
)
from verkeye.cv22.parameter_extraction import parse_fastconv_capture


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"
DVI_SHA256 = "64aa5455db12adcf3073321ebab165d177bb4956aace66a0b2d6cfe5a9568146"
ORACLE_IMAGE = (
    "xiezhouyi/cnngen@"
    "sha256:4c4266444740d9d52f6dd695fff86763ebb51dfb01b0c9778ca20209308cdcdb"
)
FASTCONV_CALLS = {0: 52, 12: 53, 25: 54}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root", type=Path, default=REPOSITORY / ".runtime/ades-full-kernels"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPOSITORY / ".runtime/ades-split9/d0-logical-from-full.bin",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-cb62/kernel-dump-run",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY / "evidence/compatibility/cavalry-yolov6n-hor-verbose.txt",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-split09-mixed-validation.json",
    )
    parser.add_argument(
        "--performance-out",
        type=Path,
        default=REPOSITORY / "evidence/performance/macos-mlx-split09.json",
    )
    return parser


def _capture(root: Path, call_index: int):
    entries = (root / f"fastconv-{call_index}-entries.bin").read_bytes()
    count = len(entries) // 96
    payloads = {
        index: path.read_bytes()
        for index in range(count)
        if (path := root / f"fastconv-{call_index}-entry-{index}-offset-0.bin").is_file()
    }
    return parse_fastconv_capture(entries, payloads)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _percentile(samples: list[int], percentile: int) -> int:
    ordered = sorted(samples)
    return ordered[max(0, (percentile * len(ordered) + 99) // 100 - 1)]


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 1 or args.frames < 1:
        raise SystemExit("warmup and frames must be positive")
    captures = {
        operator_id: _capture(args.capture_root, call_index)
        for operator_id, call_index in FASTCONV_CALLS.items()
    }

    def session(
        operator_id: int, geometry: FastconvGeometry, input_channels: int
    ) -> MlxFastconvSession:
        return MlxFastconvSession(
            captures[operator_id].channels,
            geometry,
            input_channels=input_channels,
            input_dtype="int8",
            output_dtype="int8",
        )

    sessions = {
        0: session(0, FastconvGeometry(3, 3, 1, 1, 1, 1), 64),
        12: session(12, FastconvGeometry(1, 1, 1, 1, 0, 0), 64),
        25: session(25, FastconvGeometry(1, 1, 1, 1, 0, 0), 64),
    }
    input_tensor = np.fromfile(args.input, dtype=np.int8).reshape(64, 76, 136)

    def infer() -> tuple[dict[int, np.ndarray], tuple[int, int]]:
        feature = sessions[0].infer(input_tensor)
        score_source, box_source = channel_halves(feature)
        scores = sessions[25].infer(
            quantized_silu_int8(
                score_source,
                input_exponent_offset=1,
                output_exponent_offset=2,
            )
        )
        boxes = sessions[12].infer(
            quantized_silu_int8(
                box_source,
                input_exponent_offset=1,
                output_exponent_offset=3,
            )
        )
        return (
            {
                0: boxes.astype(np.float32) / 4.0,
                1: cv22_sigmoid_float32(scores, input_exponent_offset=2),
            },
            (int(scores.min()), int(scores.max())),
        )

    manifest = parse_cavalry_verbose(args.manifest.read_text(encoding="utf-8"))
    split = manifest.splits[9]
    expected = {
        index: decode_float32_chw(
            (args.oracle_root / f"split-09-output-{index:02}.bin").read_bytes(),
            split.outputs[index],
        )[0]
        for index in range(2)
    }
    outputs, score_domain = infer()
    boundaries = []
    for index in range(2):
        actual = outputs[index]
        oracle = expected[index]
        passed = actual.shape == oracle.shape and actual.dtype == oracle.dtype and np.array_equal(actual, oracle)
        boundaries.append(
            {
                "output_index": index,
                "name": split.outputs[index].name,
                "shape": list(actual.shape),
                "dtype": actual.dtype.name,
                "size": actual.nbytes,
                "oracle_sha256": _sha256(oracle.tobytes(order="C")),
                "actual_sha256": _sha256(actual.tobytes(order="C")),
                "mismatch_count": int(np.count_nonzero(actual != oracle)),
                "comparison": "byte-identical" if passed else "different",
                "passed": passed,
            }
        )
    if not all(boundary["passed"] for boundary in boundaries):
        raise SystemExit("one or more MLX split-9 boundaries differ from ADES")

    for _ in range(args.warmup):
        infer()
    samples: list[int] = []
    started = time.perf_counter_ns()
    warmed = None
    for _ in range(args.frames):
        frame_started = time.perf_counter_ns()
        warmed, _ = infer()
        samples.append(time.perf_counter_ns() - frame_started)
    elapsed_ns = time.perf_counter_ns() - started
    assert warmed is not None
    if any(not np.array_equal(warmed[index], expected[index]) for index in expected):
        raise SystemExit("warmed split-9 output differs from ADES")

    validation = {
        "schema": "verkeye.cv22.split-mixed-validation.v1",
        "identity": {
            "model_sha256": MODEL_SHA256,
            "dvi_sha256": DVI_SHA256,
            "split_index": 9,
            "oracle_image": ORACLE_IMAGE,
            "providers": {
                str(index): dict(session.identity)
                for index, session in sessions.items()
            },
        },
        "inputs": [
            {
                "descriptor": "d0",
                "shape": list(input_tensor.shape),
                "dtype": input_tensor.dtype.name,
                "size": input_tensor.nbytes,
                "sha256": _sha256(input_tensor.tobytes(order="C")),
            }
        ],
        "semantics": {
            "operator_count": 36,
            "families": {
                "fastconvolution_operator_t": 3,
                "lvlcurve_operator_t": 3,
                "multiplyadd_operator_t": 21,
                "shuffle_operator_t": 2,
                "transcendental_operator_t": 6,
                "terminate_operator_t": 1,
            },
            "channel_assignment": "score_first_half_box_second_half",
            "score_domain": list(score_domain),
            "numeric_rules": [
                "exact recovered sparse fastconvolution with CV22 signed requantization",
                "quantized SiLU scale pairs 1->2 and 1->3",
                "CV22 unsigned-5.11 sigmoid lookup decoded to exact float32",
                "box head decodes int8/2^2 to exact float32",
            ],
        },
        "fastconv": [
            {
                "operator_id": index,
                "call_index": FASTCONV_CALLS[index],
                "parameter_entries_sha256": captures[index].entries_sha256,
                "output_channels": len(captures[index].channels),
                "kernel_point_count": sum(
                    len(channel.points) for channel in captures[index].channels
                ),
            }
            for index in FASTCONV_CALLS
        ],
        "boundaries": boundaries,
        "passed": True,
    }
    performance = {
        "schema": "verkeye.performance.v1",
        "backend": "mlx-cv22-split09",
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "elapsed_ns": elapsed_ns,
        "fps": args.frames * 1_000_000_000 / elapsed_ns,
        "samples_ns": samples,
        "summary_ns": {
            "p50": int(statistics.median(samples)),
            "p95": _percentile(samples, 95),
            "max": max(samples),
        },
        "output_sha256": {
            str(index): _sha256(value.tobytes(order="C"))
            for index, value in warmed.items()
        },
    }
    args.validation_out.parent.mkdir(parents=True, exist_ok=True)
    args.performance_out.parent.mkdir(parents=True, exist_ok=True)
    args.validation_out.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    args.performance_out.write_text(json.dumps(performance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "fps": performance["fps"], "boundaries": len(boundaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
