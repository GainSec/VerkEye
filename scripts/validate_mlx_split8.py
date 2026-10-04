#!/usr/bin/env python3
"""Prove and benchmark the recovered CB62 split-8 detector graph."""

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
DVI_SHA256 = "7cdf133f2d43e7e12b1c9524f5b8e7076f2955161150443ee0e1d9b922a25c07"
ORACLE_IMAGE = (
    "xiezhouyi/cnngen@"
    "sha256:4c4266444740d9d52f6dd695fff86763ebb51dfb01b0c9778ca20209308cdcdb"
)
FASTCONV_CALLS = {0: 44, 12: 45, 25: 46, 35: 47, 46: 48, 58: 49, 71: 50, 81: 51}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root", type=Path, default=REPOSITORY / ".runtime/ades-full-kernels"
    )
    parser.add_argument(
        "--input-root", type=Path, default=REPOSITORY / ".runtime/ades-split8"
    )
    parser.add_argument(
        "--input0",
        type=Path,
        default=REPOSITORY / ".runtime/ades-split8-input0/d0-logical-from-full.bin",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-cb62/kernel-dump-run",
    )
    parser.add_argument(
        "--internal-output",
        type=Path,
        default=REPOSITORY / ".runtime/ades-split9/d0-logical-from-full.bin",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY / "evidence/compatibility/cavalry-yolov6n-hor-verbose.txt",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-split08-mixed-validation.json",
    )
    parser.add_argument(
        "--performance-out",
        type=Path,
        default=REPOSITORY / "evidence/performance/macos-mlx-split08.json",
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
        operator_id: int,
        geometry: FastconvGeometry,
        input_channels: int,
        input_dtype: str,
    ) -> MlxFastconvSession:
        return MlxFastconvSession(
            captures[operator_id].channels,
            geometry,
            input_channels=input_channels,
            input_dtype=input_dtype,
            output_dtype="int8",
        )

    sessions = {
        0: session(0, FastconvGeometry(3, 3, 1, 1, 1, 1), 256, "int8"),
        12: session(12, FastconvGeometry(1, 1, 1, 1, 0, 0), 256, "int8"),
        25: session(25, FastconvGeometry(1, 1, 1, 1, 0, 0), 256, "int8"),
        35: session(35, FastconvGeometry(1, 1, 1, 1, 0, 0), 128, "uint8"),
        46: session(46, FastconvGeometry(3, 3, 1, 1, 1, 1), 128, "int8"),
        58: session(58, FastconvGeometry(1, 1, 1, 1, 0, 0), 128, "int8"),
        71: session(71, FastconvGeometry(1, 1, 1, 1, 0, 0), 128, "int8"),
        81: session(81, FastconvGeometry(1, 1, 1, 1, 0, 0), 64, "uint8"),
    }
    inputs = {
        0: np.fromfile(args.input0, dtype=np.int8).reshape(256, 19, 34),
        1: np.fromfile(
            args.input_root / "d1-logical-from-full.bin", dtype=np.uint8
        ).reshape(64, 76, 136),
        2: np.fromfile(
            args.input_root / "d2-logical-from-full.bin", dtype=np.uint8
        ).reshape(128, 38, 68),
    }

    def infer() -> tuple[dict[int, np.ndarray], dict[str, tuple[int, int]]]:
        low = sessions[0].infer(inputs[0])
        low_score_source, low_box_source = channel_halves(low)
        low_scores = sessions[25].infer(
            quantized_silu_int8(
                low_score_source,
                input_exponent_offset=3,
                output_exponent_offset=3,
            )
        )
        low_boxes = sessions[12].infer(quantized_silu_int8(low_box_source))

        medium_stem = quantized_silu_int8(sessions[35].infer(inputs[2]))
        medium = sessions[46].infer(medium_stem)
        medium_score_source, medium_box_source = channel_halves(medium)
        medium_scores = sessions[71].infer(
            quantized_silu_int8(
                medium_score_source,
                input_exponent_offset=1,
                output_exponent_offset=2,
            )
        )
        medium_boxes = sessions[58].infer(
            quantized_silu_int8(
                medium_box_source,
                input_exponent_offset=1,
                output_exponent_offset=3,
            )
        )

        high_feature = quantized_silu_int8(
            sessions[81].infer(inputs[1]),
            input_exponent_offset=2,
            output_exponent_offset=4,
        )
        return (
            {
                0: low_boxes.astype(np.float32) / 4.0,
                1: cv22_sigmoid_float32(
                    low_scores, input_exponent_offset=3
                ),
                2: medium_boxes.astype(np.float32) / 4.0,
                3: cv22_sigmoid_float32(
                    medium_scores, input_exponent_offset=2
                ),
                4: high_feature,
            },
            {
                "low_score_logits": (int(low_scores.min()), int(low_scores.max())),
                "medium_score_logits": (
                    int(medium_scores.min()),
                    int(medium_scores.max()),
                ),
            },
        )

    manifest = parse_cavalry_verbose(args.manifest.read_text(encoding="utf-8"))
    split = manifest.splits[8]
    expected = {
        index: decode_float32_chw(
            (args.oracle_root / f"split-08-output-{index:02}.bin").read_bytes(),
            split.outputs[index],
        )[0]
        for index in range(4)
    }
    expected[4] = np.fromfile(args.internal_output, dtype=np.int8).reshape(64, 76, 136)

    outputs, score_domains = infer()
    boundaries = []
    for index in range(5):
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
        raise SystemExit("one or more MLX split-8 boundaries differ from ADES")

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
        raise SystemExit("warmed split-8 output differs from ADES")

    validation = {
        "schema": "verkeye.cv22.split-mixed-validation.v1",
        "identity": {
            "model_sha256": MODEL_SHA256,
            "dvi_sha256": DVI_SHA256,
            "split_index": 8,
            "oracle_image": ORACLE_IMAGE,
            "providers": {
                str(index): dict(session.identity)
                for index, session in sessions.items()
            },
        },
        "inputs": [
            {
                "descriptor": f"d{index}",
                "shape": list(tensor.shape),
                "dtype": tensor.dtype.name,
                "size": tensor.nbytes,
                "sha256": _sha256(tensor.tobytes(order="C")),
            }
            for index, tensor in inputs.items()
        ],
        "semantics": {
            "operator_count": 93,
            "families": {
                "fastconvolution_operator_t": 8,
                "lvlcurve_operator_t": 8,
                "multiplyadd_operator_t": 56,
                "shuffle_operator_t": 4,
                "transcendental_operator_t": 16,
                "terminate_operator_t": 1,
            },
            "channel_assignment": "score_first_half_box_second_half",
            "score_domains": {
                name: list(domain) for name, domain in score_domains.items()
            },
            "numeric_rules": [
                "exact recovered sparse fastconvolution with CV22 signed requantization",
                "quantized SiLU scale pairs 3->3, 3->4, 1->2, 1->3, and 2->4",
                "CV22 unsigned-5.11 sigmoid lookup decoded to exact float32",
                "box heads decode int8/2^2 to exact float32",
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
        "backend": "mlx-cv22-split08",
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
