#!/usr/bin/env python3
"""Prove and benchmark the exact recovered CB62 split-6 mixed graph."""

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
    concatenate_channels,
    scale_unsigned,
)
from verkeye.cv22.parameter_extraction import parse_fastconv_capture


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"
DVI_SHA256 = "9106cd92fd02e7b2d8183dc11446f3036701a87b42eccf011ca159bfb9d9d6a4"
ORACLE_IMAGE = (
    "xiezhouyi/cnngen@"
    "sha256:4c4266444740d9d52f6dd695fff86763ebb51dfb01b0c9778ca20209308cdcdb"
)
OUTPUT_FORMATS = {
    0: ((192, 76, 136), "int8", 4),
    1: ((64, 76, 136), "uint8", 5),
    2: ((64, 76, 136), "uint8", 5),
    3: ((64, 76, 136), "uint8", 5),
    4: ((64, 76, 136), "uint8", 4),
    5: ((64, 38, 68), "uint8", 3),
    6: ((64, 38, 68), "uint8", 3),
    7: ((82, 38, 68), "uint16", 10),
    8: ((82, 38, 68), "uint16", 11),
    9: ((75, 38, 68), "uint16", 11),
    10: ((82, 38, 68), "uint16", 11),
    11: ((128, 38, 68), "uint8", 4),
    12: ((128, 19, 34), "uint8", 4),
    13: ((128, 19, 34), "uint8", 4),
    14: ((251, 19, 34), "uint8", 5),
}
FASTCONV_CALLS = {
    1: 29,
    2: 30,
    3: 31,
    4: 32,
    5: 33,
    7: 34,
    9: 35,
    10: 36,
    11: 37,
    12: 38,
    14: 39,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root", type=Path, default=REPOSITORY / ".runtime/ades-full-kernels"
    )
    parser.add_argument(
        "--oracle-root", type=Path, default=REPOSITORY / ".runtime/ades-split6"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-split06-mixed-validation.json",
    )
    parser.add_argument(
        "--performance-out",
        type=Path,
        default=REPOSITORY / "evidence/performance/macos-mlx-split06.json",
    )
    return parser


def _capture(root: Path, call_index: int):
    entries = (root / f"fastconv-{call_index}-entries.bin").read_bytes()
    count = len(entries) // 96
    payloads = {}
    for index in range(count):
        path = root / f"fastconv-{call_index}-entry-{index}-offset-0.bin"
        if path.is_file():
            payloads[index] = path.read_bytes()
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
        1: MlxFastconvSession(
            captures[1].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=192,
            input_dtype="int8",
        ),
        2: MlxFastconvSession(
            captures[2].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=64,
        ),
        3: MlxFastconvSession(
            captures[3].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=64,
        ),
        4: MlxFastconvSession(
            captures[4].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=64,
        ),
        5: MlxFastconvSession(
            captures[5].channels,
            FastconvGeometry(3, 3, 2, 2, 1, 1),
            input_channels=64,
        ),
        7: MlxFastconvSession(
            captures[7].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=128,
            output_dtype="uint16",
        ),
        9: MlxFastconvSession(
            captures[9].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=82,
            input_dtype="uint16",
            output_dtype="uint16",
        ),
        10: MlxFastconvSession(
            captures[10].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=75,
            input_dtype="uint16",
            output_dtype="uint16",
        ),
        11: MlxFastconvSession(
            captures[11].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=82,
            input_dtype="uint16",
        ),
        12: MlxFastconvSession(
            captures[12].channels,
            FastconvGeometry(3, 3, 2, 2, 1, 1),
            input_channels=128,
        ),
        14: MlxFastconvSession(
            captures[14].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=256,
        ),
    }

    input_paths = {
        name: args.oracle_root / f"{name}-logical.bin"
        for name in ("d0", "d1", "d2", "d3")
    }
    input_bytes = {name: path.read_bytes() for name, path in input_paths.items()}
    d0 = np.frombuffer(input_bytes["d0"], dtype=np.uint8).reshape(64, 38, 68)
    d1 = np.frombuffer(input_bytes["d1"], dtype=np.int8).reshape(64, 76, 136)
    d2 = np.frombuffer(input_bytes["d2"], dtype=np.int8).reshape(128, 76, 136)
    d3 = np.frombuffer(input_bytes["d3"], dtype=np.uint8).reshape(128, 19, 34)

    def infer() -> dict[int, np.ndarray]:
        output: dict[int, np.ndarray] = {}
        output[0] = concatenate_channels(d1, d2)
        output[1] = sessions[1].infer(output[0])
        output[2] = sessions[2].infer(output[1])
        output[3] = sessions[3].infer(output[2])
        output[4] = sessions[4].infer(output[3])
        output[5] = sessions[5].infer(output[4])
        output[6] = d0.copy()
        d28 = concatenate_channels(output[5], output[6])
        output[7] = sessions[7].infer(d28)
        output[8] = scale_unsigned(
            output[7], multiplier=2, divisor=1, output_dtype="uint16"
        )
        output[9] = sessions[9].infer(output[8])
        output[10] = sessions[10].infer(output[9])
        output[11] = sessions[11].infer(output[10])
        output[12] = sessions[12].infer(output[11])
        output[13] = d3.copy()
        d27 = concatenate_channels(output[12], output[13])
        output[14] = sessions[14].infer(d27)
        return output

    outputs = infer()
    boundaries = []
    for operator_id in range(15):
        expected_shape, expected_dtype, exponent_offset = OUTPUT_FORMATS[operator_id]
        actual = outputs[operator_id]
        oracle_member = (
            f"op{operator_id}a.bin"
            if operator_id in {6, 13}
            else f"op{operator_id}.bin"
        )
        oracle_path = args.oracle_root / oracle_member
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
                0: "merge_operator_t",
                6: "shuffle_operator_t",
                8: "multiplyadd_operator_t",
                13: "shuffle_operator_t",
            }[operator_id]
        )
        boundary = {
            "operator_id": operator_id,
            "operator_type": operator_type,
            "shape": list(expected_shape),
            "dtype": expected_dtype,
            "exponent_offset": exponent_offset,
            "size": len(oracle_bytes),
            "oracle_member": oracle_member,
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
        raise SystemExit("one or more MLX split-6 boundaries differ from ADES")

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
        != (
            args.oracle_root
            / (f"op{index}a.bin" if index in {6, 13} else f"op{index}.bin")
        ).read_bytes()
        for index in range(15)
    ):
        raise SystemExit("warmed MLX split-6 output differs from ADES")

    full_output_checks = {
        "split_output_0": {
            "operator_id": 4,
            "full_runtime_logical_member": "../ades-split8/d1-logical-from-full.bin",
        },
        "split_output_1": {
            "operator_id": 11,
            "full_runtime_logical_member": "../ades-split8/d2-logical-from-full.bin",
        },
        "split_output_2": {
            "operator_id": 14,
            "full_runtime_logical_member": "../ades-split7/d0-logical-from-full.bin",
        },
    }
    for check in full_output_checks.values():
        path = args.oracle_root / str(check["full_runtime_logical_member"])
        expected = path.read_bytes()
        actual = warmed[int(check["operator_id"])].tobytes(order="C")
        check.update(
            {
                "size": len(expected),
                "oracle_sha256": _sha256(expected),
                "actual_sha256": _sha256(actual),
                "comparison": "byte-identical" if actual == expected else "different",
                "passed": actual == expected,
            }
        )
    if not all(bool(check["passed"]) for check in full_output_checks.values()):
        raise SystemExit("split-6 declared output differs from full-runtime dump")

    validation = {
        "schema": "verkeye.cv22.split-mixed-validation.v1",
        "identity": {
            "model_sha256": MODEL_SHA256,
            "dvi_sha256": DVI_SHA256,
            "split_index": 6,
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
            "operator_count": 15,
            "families": {
                "fastconvolution_operator_t": 11,
                "merge_operator_t": 1,
                "multiplyadd_operator_t": 1,
                "shuffle_operator_t": 2,
            },
            "numeric_rules": [
                "merge concatenates channels while preserving signed int8 values",
                "shuffle mode zero is a byte-preserving copy",
                "unsigned multiplyadd doubles without rounding",
                "stride-two 3x3 fastconvolution uses one-cell symmetric padding",
                "d28 alias is depth concat(op5, op6)",
                "d27 alias is depth concat(op12, op13)",
            ],
            "alias_reconstruction": {
                "d28": ["op5", "op6"],
                "d27": ["op12", "op13"],
                "verification": "all declared split outputs match full-runtime dumps",
            },
        },
        "boundaries": boundaries,
        "full_runtime_output_checks": full_output_checks,
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
            "split_index": 6,
            "operator_count": 15,
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
                "full_runtime_outputs_passed": all(
                    bool(check["passed"]) for check in full_output_checks.values()
                ),
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
