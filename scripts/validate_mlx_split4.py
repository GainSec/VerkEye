#!/usr/bin/env python3
"""Prove and benchmark the exact recovered CB62 split-4 mixed graph."""

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
    channel_halves,
    concatenate_channels,
    max_filter_5x5,
    nearest_resample_2x,
    requantize_signed,
)
from verkeye.cv22.parameter_extraction import parse_fastconv_capture


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"
DVI_SHA256 = "eaebd733b9d8a509ed5387341043b719ca49eb69bacdfceadaaca77d52336132"
ORACLE_IMAGE = (
    "xiezhouyi/cnngen@"
    "sha256:4c4266444740d9d52f6dd695fff86763ebb51dfb01b0c9778ca20209308cdcdb"
)
OUTPUT_FORMATS = {
    0: ((256, 19, 34), "uint8", 6),
    **{index: ((128, 19, 34), "uint8", 6) for index in range(1, 13)},
    13: ((256, 19, 34), "uint8", 6),
    14: ((512, 19, 34), "uint8", 6),
    15: ((512, 19, 34), "uint8", 6),
    16: ((1024, 19, 34), "uint8", 6),
    17: ((507, 19, 34), "uint8", 5),
    18: ((128, 19, 34), "uint8", 4),
    19: ((128, 38, 68), "uint8", 4),
    20: ((128, 38, 68), "uint8", 4),
    21: ((128, 38, 68), "int8", 8),
    22: ((128, 38, 68), "int8", 5),
    23: ((256, 38, 68), "int8", 5),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root", type=Path, default=REPOSITORY / ".runtime/ades-full-kernels"
    )
    parser.add_argument(
        "--oracle-root", type=Path, default=REPOSITORY / ".runtime/ades-split4"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-split04-mixed-validation.json",
    )
    parser.add_argument(
        "--performance-out",
        type=Path,
        default=REPOSITORY / "evidence/performance/macos-mlx-split04.json",
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

    captures = {index: _capture(args.capture_root, index) for index in range(19, 23)}
    sessions = {
        0: MlxFastconvSession(
            captures[19].channels,
            FastconvGeometry(1, 1, 1, 1, 0, 0),
            input_channels=83,
            input_dtype="uint16",
        ),
        17: MlxFastconvSession(
            captures[20].channels,
            FastconvGeometry(1, 1, 1, 1, 0, 0),
            input_channels=1024,
        ),
        18: MlxFastconvSession(
            captures[21].channels,
            FastconvGeometry(1, 1, 1, 1, 0, 0),
            input_channels=507,
        ),
        21: MlxFastconvSession(
            captures[22].channels,
            FastconvGeometry(
                2, 2, 1, 1, 1, 1, padding_bottom=0, padding_right=0
            ),
            input_channels=128,
            output_dtype="int8",
        ),
    }

    input_paths = {
        "d0": args.oracle_root / "d0-logical.bin",
        "d1": REPOSITORY / ".runtime/ades-split3/op3.bin",
        "d15": args.oracle_root / "d15.bin",
        "d16": args.oracle_root / "d16.bin",
        "d9": args.oracle_root / "d9-logical.bin",
    }
    input_bytes = {name: path.read_bytes() for name, path in input_paths.items()}
    d0 = np.frombuffer(input_bytes["d0"], dtype=np.uint16).reshape(256, 38, 68)
    d1 = np.frombuffer(input_bytes["d1"], dtype=np.uint16).reshape(83, 19, 34)
    d15 = np.frombuffer(input_bytes["d15"], dtype=np.uint8).reshape(256, 19, 34)
    d16 = np.frombuffer(input_bytes["d16"], dtype=np.uint8).reshape(256, 19, 34)
    mux_mask = input_bytes["d9"]

    def infer() -> dict[int, np.ndarray]:
        output: dict[int, np.ndarray] = {}
        output[0] = sessions[0].infer(d1)
        output[1], output[2] = channel_halves(output[0])
        output[3] = max_filter_5x5(output[1])
        output[4] = max_filter_5x5(output[2])
        output[5], output[7] = channel_halves(d15)
        output[6] = max_filter_5x5(output[5])
        output[8] = max_filter_5x5(output[7])
        output[9], output[11] = channel_halves(d16)
        output[10] = max_filter_5x5(output[9])
        output[12] = max_filter_5x5(output[11])
        output[13] = concatenate_channels(output[10], output[12])
        output[14] = concatenate_channels(output[0], d15)
        output[15] = concatenate_channels(d16, output[13])
        output[16] = concatenate_channels(output[14], output[15])
        output[17] = sessions[17].infer(output[16])
        output[18] = sessions[18].infer(output[17])
        output[19] = nearest_resample_2x(output[18])
        output[20] = apply_bitpacked_mux_mask(output[19], mux_mask)
        output[21] = sessions[21].infer(output[20])
        output[22] = requantize_signed(output[21], divisor=8)
        output[23] = requantize_signed(d0, divisor=128)
        return output

    outputs = infer()
    boundaries = []
    for operator_id in range(24):
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
        boundaries.append(
            {
                "operator_id": operator_id,
                "operator_type": (
                    "fastconvolution_operator_t"
                    if operator_id in sessions
                    else {
                        **{index: "shuffle_operator_t" for index in (1, 2, 5, 7, 9, 11)},
                        **{index: "minmax_operator_t" for index in (3, 4, 6, 8, 10, 12)},
                        **{index: "merge_operator_t" for index in (13, 14, 15, 16)},
                        19: "resample_operator_t",
                        20: "mux_operator_t",
                        22: "multiplyadd_operator_t",
                        23: "multiplyadd_operator_t",
                    }[operator_id]
                ),
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
                **(
                    {
                        "fastconv_call_index": {
                            0: 19,
                            17: 20,
                            18: 21,
                            21: 22,
                        }[operator_id],
                        "parameter_entries_sha256": captures[
                            {0: 19, 17: 20, 18: 21, 21: 22}[operator_id]
                        ].entries_sha256,
                        "kernel_point_count": sum(
                            len(channel.points)
                            for channel in captures[
                                {0: 19, 17: 20, 18: 21, 21: 22}[operator_id]
                            ].channels
                        ),
                    }
                    if operator_id in sessions
                    else {}
                ),
            }
        )
    if not all(boundary["passed"] for boundary in boundaries):
        raise SystemExit("one or more MLX split-4 boundaries differ from ADES")

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
        for index in range(24)
    ):
        raise SystemExit("warmed MLX split-4 output differs from ADES")

    validation = {
        "schema": "verkeye.cv22.split-mixed-validation.v1",
        "identity": {
            "model_sha256": MODEL_SHA256,
            "dvi_sha256": DVI_SHA256,
            "split_index": 4,
            "oracle_image": ORACLE_IMAGE,
            "oracle_run_log_sha256": _sha256(
                (args.oracle_root / "run-full.log").read_bytes()
            ),
            "providers": {
                str(index): dict(session.identity) for index, session in sessions.items()
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
            "operator_count": 24,
            "families": {
                "fastconvolution_operator_t": 4,
                "shuffle_operator_t": 6,
                "minmax_operator_t": 6,
                "merge_operator_t": 4,
                "resample_operator_t": 1,
                "mux_operator_t": 1,
                "multiplyadd_operator_t": 2,
            },
            "numeric_rules": [
                "channel-preserving first/second-half shuffle",
                "5x5 maximum with two-cell zero padding",
                "depth-axis concatenation",
                "two-times nearest-neighbor spatial resample",
                "row-aligned little-endian 1-bit select-input-or-zero mux",
                "signed half-away-from-zero requantization",
                "2x2 fastconvolution uses top/left-only one-cell padding",
                "signed fastconvolution output uses (value + 2) arithmetic shift by two",
            ],
            "boundary_adapter_status": (
                "d15/d16 logical alias views are exact ADES-extracted inputs; "
                "the portable split-3-to-split-4 packed boundary adapter remains pending"
            ),
        },
        "boundaries": boundaries,
        "result": {
            "comparison": "byte-identical at every non-terminate operator boundary",
            "compared_bytes": sum(boundary["size"] for boundary in boundaries),
            "mismatch_count": sum(boundary["mismatch_count"] for boundary in boundaries),
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
            "split_index": 4,
            "operator_count": 24,
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
        "inference_count": sum(session.inference_count for session in sessions.values()),
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
