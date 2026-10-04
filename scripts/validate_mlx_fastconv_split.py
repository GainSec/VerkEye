#!/usr/bin/env python3
"""Validate and benchmark a recovered CB62 all-fastconv split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np

from verkeye.accelerated.macos import MlxFastconvChain, MlxFastconvSession
from verkeye.cv22.operators import FastconvGeometry
from verkeye.cv22.parameter_extraction import parse_fastconv_capture


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"
ORACLE_IMAGE = (
    "xiezhouyi/cnngen@"
    "sha256:4c4266444740d9d52f6dd695fff86763ebb51dfb01b0c9778ca20209308cdcdb"
)
SPLITS = {
    0: {
        "capture_root": REPOSITORY / ".runtime/ades-direct",
        "oracle_root": REPOSITORY / ".runtime/ades-direct",
        "input": REPOSITORY / ".runtime/ades-direct/d12.bin",
        "input_shape": (3, 608, 1088),
        "input_dtype": "uint8",
        "input_exp_offset": 8,
        "call_indices": (0, 1, 2, 3, 4, 5),
        "input_channels": (3, 22, 64, 64, 64, 128),
        "strides": (2, 2, 1, 1, 2, 1),
        "output_shapes": (
            (22, 304, 544),
            (64, 152, 272),
            (64, 152, 272),
            (64, 152, 272),
            (128, 76, 136),
            (127, 76, 136),
        ),
        "output_dtypes": ("uint8",) * 6,
        "output_exp_offsets": (4, 4, 4, 4, 4, 5),
    },
    1: {
        "capture_root": REPOSITORY / ".runtime/ades-full-kernels",
        "oracle_root": REPOSITORY / ".runtime/ades-split1",
        "input": REPOSITORY / ".runtime/ades-direct/op5.bin",
        "input_shape": (127, 76, 136),
        "input_dtype": "uint8",
        "input_exp_offset": 5,
        "call_indices": (6, 7, 8, 9, 10),
        "input_channels": (127, 128, 128, 128, 158),
        "strides": (1, 1, 1, 2, 1),
        "output_shapes": (
            (128, 76, 136),
            (128, 76, 136),
            (128, 76, 136),
            (158, 38, 68),
            (168, 38, 68),
        ),
        "output_dtypes": ("uint8", "uint8", "uint8", "uint16", "uint16"),
        "output_exp_offsets": (5, 5, 5, 10, 10),
    },
    2: {
        "capture_root": REPOSITORY / ".runtime/ades-full-kernels",
        "oracle_root": REPOSITORY / ".runtime/ades-split2",
        "input": REPOSITORY / ".runtime/ades-split1/op4.bin",
        "input_shape": (168, 38, 68),
        "input_dtype": "uint16",
        "input_exp_offset": 10,
        "call_indices": (11, 12, 13, 14),
        "input_channels": (168, 155, 161, 173),
        "strides": (1, 1, 1, 1),
        "output_shapes": (
            (155, 38, 68),
            (161, 38, 68),
            (173, 38, 68),
            (186, 38, 68),
        ),
        "output_dtypes": ("uint16",) * 4,
        "output_exp_offsets": (10, 10, 11, 11),
    },
    3: {
        "capture_root": REPOSITORY / ".runtime/ades-full-kernels",
        "oracle_root": REPOSITORY / ".runtime/ades-split3",
        "input": REPOSITORY / ".runtime/ades-split2/op3.bin",
        "input_shape": (186, 38, 68),
        "input_dtype": "uint16",
        "input_exp_offset": 11,
        "call_indices": (15, 16, 17, 18),
        "input_channels": (186, 256, 81, 483),
        "strides": (1, 2, 1, 1),
        "output_shapes": (
            (256, 38, 68),
            (81, 19, 34),
            (483, 19, 34),
            (83, 19, 34),
        ),
        "output_dtypes": ("uint16", "uint16", "uint8", "uint16"),
        "output_exp_offsets": (12, 11, 5, 13),
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-index", type=int, choices=tuple(SPLITS), required=True)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--validation-out",
        type=Path,
    )
    parser.add_argument(
        "--performance-out",
        type=Path,
    )
    return parser


def _load_capture(root: Path, call_index: int):
    entries = (root / f"fastconv-{call_index}-entries.bin").read_bytes()
    channel_count = len(entries) // 96
    payloads = {}
    for channel_index in range(channel_count):
        path = root / f"fastconv-{call_index}-entry-{channel_index}-offset-0.bin"
        if path.is_file():
            payloads[channel_index] = path.read_bytes()
    return parse_fastconv_capture(entries, payloads)


def _percentile(samples: list[int], percentile: int) -> int:
    ordered = sorted(samples)
    return ordered[max(0, (percentile * len(ordered) + 99) // 100 - 1)]


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 1 or args.frames < 1:
        raise SystemExit("warmup and frames must be positive")
    spec = SPLITS[args.split_index]
    capture_root = args.capture_root or spec["capture_root"]
    oracle_root = args.oracle_root or spec["oracle_root"]
    input_path = args.input or spec["input"]
    validation_out = args.validation_out or (
        REPOSITORY
        / f"evidence/accelerated/cb62-split{args.split_index:02d}-fastconv-validation.json"
    )
    performance_out = args.performance_out or (
        REPOSITORY
        / f"evidence/performance/macos-mlx-split{args.split_index:02d}.json"
    )

    sessions = []
    captures = []
    for call_index, input_channels, stride, input_dtype in zip(
        spec["call_indices"],
        spec["input_channels"],
        spec["strides"],
        (spec["input_dtype"], *spec["output_dtypes"][:-1]),
        strict=True,
    ):
        capture = _load_capture(capture_root, call_index)
        captures.append(capture)
        sessions.append(
            MlxFastconvSession(
                capture.channels,
                FastconvGeometry(3, 3, stride, stride, 1, 1),
                input_channels=input_channels,
                input_dtype=input_dtype,
            )
        )
    chain = MlxFastconvChain(tuple(sessions))

    input_bytes = input_path.read_bytes()
    input_tensor = np.frombuffer(
        input_bytes, dtype=np.dtype(spec["input_dtype"])
    ).reshape(spec["input_shape"])
    actual_outputs = chain.infer_with_intermediates(input_tensor)
    boundaries = []
    all_passed = True
    for operator_id, (
        call_index,
        actual,
        expected_shape,
        expected_dtype,
        exponent_offset,
        capture,
    ) in enumerate(
        zip(
            spec["call_indices"],
            actual_outputs,
            spec["output_shapes"],
            spec["output_dtypes"],
            spec["output_exp_offsets"],
            captures,
            strict=True,
        )
    ):
        oracle_path = oracle_root / f"op{operator_id}.bin"
        oracle_bytes = oracle_path.read_bytes()
        actual_bytes = actual.tobytes(order="C")
        oracle = np.frombuffer(
            oracle_bytes, dtype=np.dtype(expected_dtype)
        ).reshape(expected_shape)
        mismatch_count = int(np.count_nonzero(actual != oracle))
        passed = actual.shape == expected_shape and actual_bytes == oracle_bytes
        all_passed = all_passed and passed
        boundaries.append(
            {
                "operator_id": operator_id,
                "call_index": call_index,
                "shape": list(expected_shape),
                "dtype": expected_dtype,
                "exponent_offset": exponent_offset,
                "oracle_member": oracle_path.name,
                "size": len(oracle_bytes),
                "oracle_sha256": hashlib.sha256(oracle_bytes).hexdigest(),
                "actual_sha256": hashlib.sha256(actual_bytes).hexdigest(),
                "mismatch_count": mismatch_count,
                "comparison": "byte-identical" if passed else "different",
                "passed": passed,
                "parameter_entries_sha256": capture.entries_sha256,
                "output_channel_count": len(capture.channels),
                "kernel_point_count": sum(
                    len(channel.points) for channel in capture.channels
                ),
            }
        )
    if not all_passed:
        raise SystemExit(
            f"one or more MLX split-{args.split_index} boundaries differ from ADES"
        )

    for _ in range(args.warmup):
        chain.infer(input_tensor)
    samples: list[int] = []
    started = time.perf_counter_ns()
    final = None
    for _ in range(args.frames):
        frame_started = time.perf_counter_ns()
        final = chain.infer(input_tensor)
        samples.append(time.perf_counter_ns() - frame_started)
    elapsed_ns = time.perf_counter_ns() - started
    assert final is not None
    final_oracle = oracle_root / f"op{len(sessions) - 1}.bin"
    if final.tobytes(order="C") != final_oracle.read_bytes():
        raise SystemExit(
            f"warmed MLX split-{args.split_index} output differs from ADES"
        )

    validation = {
        "schema": "verkeye.cv22.split-fastconv-validation.v1",
        "identity": {
            "model_sha256": MODEL_SHA256,
            "split_index": args.split_index,
            "oracle_image": ORACLE_IMAGE,
            "provider": dict(chain.identity),
        },
        "input": {
            "member": input_path.name,
            "shape": list(spec["input_shape"]),
            "dtype": spec["input_dtype"],
            "exponent_offset": spec["input_exp_offset"],
            "size": len(input_bytes),
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
        },
        "semantics": {
            "operator_family": "fastconvolution_operator_t",
            "operator_count": len(sessions),
            "geometry": [
                {
                    "kernel": [3, 3],
                    "stride": [stride, stride],
                    "padding": [1, 1],
                }
                for stride in spec["strides"]
            ],
            "numeric_rules": [
                "signed sparse integer correlation",
                "zero padding",
                "arithmetic per-channel accumulator shift",
                "upper saturation at twice the output maximum plus one after accumulator shift",
                "signed per-channel offset",
                "one-bit rounded right shift for final_shift_control=2",
                "declared unsigned output saturation",
            ],
        },
        "boundaries": boundaries,
        "result": {
            "comparison": "byte-identical at every operator boundary",
            "compared_bytes": sum(boundary["size"] for boundary in boundaries),
            "mismatch_count": sum(
                boundary["mismatch_count"] for boundary in boundaries
            ),
            "passed": all_passed,
        },
    }
    performance = {
        "schema": "verkeye.performance.macos-mlx-split.v1",
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "provider": dict(chain.identity),
        "scope": {
            "model_sha256": MODEL_SHA256,
            "split_index": args.split_index,
            "operator_count": len(sessions),
            "input_shape": list(spec["input_shape"]),
            "output_shape": list(spec["output_shapes"][-1]),
            "comparison": "byte-identical at every operator boundary",
            "validation_sha256": hashlib.sha256(
                (json.dumps(validation, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            ).hexdigest(),
        },
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "elapsed_ns": elapsed_ns,
        "fps": args.frames * 1_000_000_000 / elapsed_ns,
        "latency_ns": {
            "raw": samples,
            "p50": int(statistics.median(samples)),
            "p95": _percentile(samples, 95),
            "max": max(samples),
        },
        "session_creation_count": chain.session_creation_count,
        "inference_count": chain.inference_count,
    }
    _atomic_json(validation_out, validation)
    _atomic_json(performance_out, performance)
    print(
        json.dumps(
            {
                "compared_bytes": validation["result"]["compared_bytes"],
                "mismatch_count": validation["result"]["mismatch_count"],
                "fps": performance["fps"],
                "p95_ns": performance["latency_ns"]["p95"],
            },
            sort_keys=True,
        )
    )
    return 0


def _atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
