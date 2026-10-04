#!/usr/bin/env python3
"""Prove and benchmark the exact recovered CB62 split-5 mixed graph."""

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

from verkeye.accelerated.macos import MlxFastconvSession
from verkeye.cv22.operators import (
    FastconvGeometry,
    apply_bitpacked_mux_mask,
    nearest_resample_2x,
    scale_unsigned,
)
from verkeye.cv22.parameter_extraction import parse_fastconv_capture


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"
DVI_SHA256 = "3fc3806e8238912492471d8a9da2edbfdbf3f38cca4d013d56fd391b1ff4746e"
ORACLE_IMAGE = (
    "xiezhouyi/cnngen@"
    "sha256:4c4266444740d9d52f6dd695fff86763ebb51dfb01b0c9778ca20209308cdcdb"
)
OUTPUT_FORMATS = {
    0: ((115, 38, 68), "uint16", 9),
    1: ((115, 38, 68), "uint16", 11),
    2: ((115, 38, 68), "uint8", 4),
    3: ((116, 38, 68), "uint16", 11),
    4: ((102, 38, 68), "uint16", 12),
    5: ((64, 38, 68), "uint8", 3),
    6: ((64, 76, 136), "uint8", 3),
    7: ((64, 76, 136), "uint8", 3),
    8: ((64, 76, 136), "int8", 4),
    9: ((128, 76, 136), "int8", 4),
}
FASTCONV_CALLS = {0: 23, 2: 24, 3: 25, 4: 26, 5: 27, 8: 28}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root", type=Path, default=REPOSITORY / ".runtime/ades-full-kernels"
    )
    parser.add_argument(
        "--oracle-root", type=Path, default=REPOSITORY / ".runtime/ades-split5"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-split05-mixed-validation.json",
    )
    parser.add_argument(
        "--performance-out",
        type=Path,
        default=REPOSITORY / "evidence/performance/macos-mlx-split05.json",
    )
    return parser


def _capture(root: Path, call_index: int):
    entries = (root / f"fastconv-{call_index}-entries.bin").read_bytes()
    count = len(entries) // 96
    payloads = {
        index: (root / f"fastconv-{call_index}-entry-{index}-offset-0.bin").read_bytes()
        for index in range(count)
    }
    return parse_fastconv_capture(entries, payloads)


def _percentile(samples: list[int], percentile: int) -> int:
    ordered = sorted(samples)
    return ordered[max(0, (percentile * len(ordered) + 99) // 100 - 1)]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 1 or args.frames < 1:
        raise SystemExit("warmup and frames must be positive")

    captures = {
        operator_id: _capture(args.capture_root, call_index)
        for operator_id, call_index in FASTCONV_CALLS.items()
    }
    sessions = {
        0: MlxFastconvSession(
            captures[0].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=384,
            input_dtype="int8",
            output_dtype="uint16",
        ),
        2: MlxFastconvSession(
            captures[2].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=115,
            input_dtype="uint16",
        ),
        3: MlxFastconvSession(
            captures[3].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=115,
            output_dtype="uint16",
        ),
        4: MlxFastconvSession(
            captures[4].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=116,
            input_dtype="uint16",
            output_dtype="uint16",
        ),
        5: MlxFastconvSession(
            captures[5].channels,
            FastconvGeometry(1, 1, 1, 1, 0, 0),
            input_channels=102,
            input_dtype="uint16",
        ),
        8: MlxFastconvSession(
            captures[8].channels,
            FastconvGeometry(
                2, 2, 1, 1, 1, 1, padding_bottom=0, padding_right=0
            ),
            input_channels=64,
            output_dtype="int8",
        ),
    }

    input_paths = {
        "d0": args.oracle_root / "d0-logical.bin",
        "d1": args.oracle_root / "d1-logical.bin",
        "d14": args.oracle_root / "d14-logical.bin",
    }
    input_bytes = {name: path.read_bytes() for name, path in input_paths.items()}
    d0 = np.frombuffer(input_bytes["d0"], dtype=np.int8).reshape(384, 38, 68)
    d1 = np.frombuffer(input_bytes["d1"], dtype=np.uint8).reshape(128, 76, 136)
    mux_mask = input_bytes["d14"]

    def infer() -> dict[int, np.ndarray]:
        output: dict[int, np.ndarray] = {}
        output[0] = sessions[0].infer(d0)
        output[1] = scale_unsigned(
            output[0], multiplier=4, divisor=1, output_dtype="uint16"
        )
        output[2] = sessions[2].infer(output[1])
        output[3] = sessions[3].infer(output[2])
        output[4] = sessions[4].infer(output[3])
        output[5] = sessions[5].infer(output[4])
        output[6] = nearest_resample_2x(output[5])
        output[7] = apply_bitpacked_mux_mask(output[6], mux_mask)
        output[8] = sessions[8].infer(output[7])
        output[9] = scale_unsigned(
            d1, multiplier=1, divisor=2, output_dtype="int8"
        )
        return output

    outputs = infer()
    boundaries = []
    for operator_id in range(10):
        expected_shape, expected_dtype, exponent_offset = OUTPUT_FORMATS[operator_id]
        actual = outputs[operator_id]
        oracle_path = args.oracle_root / f"op{operator_id}.bin"
        oracle_bytes = oracle_path.read_bytes()
        actual_bytes = actual.tobytes(order="C")
        oracle = np.frombuffer(oracle_bytes, dtype=np.dtype(expected_dtype)).reshape(
            expected_shape
        )
        mismatch_count = int(np.count_nonzero(actual != oracle))
        passed = (
            actual.shape == expected_shape
            and actual.dtype == np.dtype(expected_dtype)
            and actual_bytes == oracle_bytes
        )
        operator_type = (
            "fastconvolution_operator_t"
            if operator_id in sessions
            else {
                1: "multiplyadd_operator_t",
                6: "resample_operator_t",
                7: "mux_operator_t",
                9: "multiplyadd_operator_t",
            }[operator_id]
        )
        boundary = {
            "operator_id": operator_id,
            "operator_type": operator_type,
            "shape": list(expected_shape),
            "dtype": expected_dtype,
            "exponent_offset": exponent_offset,
            "size": len(oracle_bytes),
            "oracle_member": oracle_path.name,
            "oracle_sha256": _sha256(oracle_bytes),
            "actual_sha256": _sha256(actual_bytes),
            "mismatch_count": mismatch_count,
            "comparison": "byte-identical" if passed else "different",
            "passed": passed,
        }
        if operator_id in sessions:
            boundary.update(
                {
                    "fastconv_call_index": FASTCONV_CALLS[operator_id],
                    "parameter_entries_sha256": captures[
                        operator_id
                    ].entries_sha256,
                    "kernel_point_count": sum(
                        len(channel.points)
                        for channel in captures[operator_id].channels
                    ),
                }
            )
        boundaries.append(boundary)
    if not all(boundary["passed"] for boundary in boundaries):
        raise SystemExit("one or more MLX split-5 boundaries differ from ADES")

    for _ in range(args.warmup):
        infer()
    samples: list[int] = []
    started = time.perf_counter_ns()
    warmed = None
    for _ in range(args.frames):
        frame_started = time.perf_counter_ns()
        warmed = infer()
        samples.append(time.perf_counter_ns() - frame_started)
    elapsed_ns = time.perf_counter_ns() - started
    assert warmed is not None
    if any(
        warmed[index].tobytes(order="C")
        != (args.oracle_root / f"op{index}.bin").read_bytes()
        for index in range(10)
    ):
        raise SystemExit("warmed MLX split-5 output differs from ADES")

    validation = {
        "schema": "verkeye.cv22.split-mixed-validation.v1",
        "identity": {
            "model_sha256": MODEL_SHA256,
            "dvi_sha256": DVI_SHA256,
            "split_index": 5,
            "oracle_image": ORACLE_IMAGE,
            "oracle_run_log_sha256": _sha256(
                (args.oracle_root / "run-full.log").read_bytes()
            ),
            "providers": {
                str(index): dict(session.identity)
                for index, session in sessions.items()
            },
        },
        "inputs": [
            {
                "descriptor": name,
                "member": path.name,
                "size": len(input_bytes[name]),
                "sha256": _sha256(input_bytes[name]),
            }
            for name, path in input_paths.items()
        ],
        "semantics": {
            "operator_count": 10,
            "families": {
                "fastconvolution_operator_t": 6,
                "multiplyadd_operator_t": 2,
                "resample_operator_t": 1,
                "mux_operator_t": 1,
            },
            "numeric_rules": [
                "signed-int8 fastconvolution input is accumulated without reinterpretation",
                "unsigned multiplyadd scales by four without rounding",
                "unsigned multiplyadd halves with round-half-up saturation to signed int8",
                "two-times nearest-neighbor spatial resample",
                "row-aligned little-endian 1-bit select-input-or-zero mux",
                "2x2 fastconvolution uses top/left-only one-cell padding",
                "signed fastconvolution output uses (value + 2) arithmetic shift by two",
            ],
        },
        "boundaries": boundaries,
        "result": {
            "comparison": "byte-identical at every non-terminate operator boundary",
            "compared_bytes": sum(boundary["size"] for boundary in boundaries),
            "mismatch_count": sum(
                boundary["mismatch_count"] for boundary in boundaries
            ),
            "passed": True,
        },
    }
    validation_bytes = (json.dumps(validation, indent=2, sort_keys=True) + "\n").encode()
    performance = {
        "schema": "verkeye.performance.macos-mlx-split.v1",
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "scope": {
            "model_sha256": MODEL_SHA256,
            "split_index": 5,
            "operator_count": 10,
            "comparison": "byte-identical at every non-terminate operator boundary",
            "validation_sha256": _sha256(validation_bytes),
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
        "session_creation_count": sum(
            session.session_creation_count for session in sessions.values()
        ),
        "inference_count": sum(
            session.inference_count for session in sessions.values()
        ),
    }
    _atomic_write(args.validation_out, validation_bytes)
    _atomic_write(
        args.performance_out,
        (json.dumps(performance, indent=2, sort_keys=True) + "\n").encode(),
    )
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


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
