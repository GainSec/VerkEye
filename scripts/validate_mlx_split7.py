#!/usr/bin/env python3
"""Prove and benchmark the recovered CB62 split-7 SiLU graph."""

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
from verkeye.cv22.operators import FastconvGeometry, quantized_silu_int8
from verkeye.cv22.parameter_extraction import parse_fastconv_capture


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"
DVI_SHA256 = "026bd75dc2413c04f00747518a2f44356e8004c00b0385614aacdc05374acbe7"
ORACLE_IMAGE = (
    "xiezhouyi/cnngen@"
    "sha256:4c4266444740d9d52f6dd695fff86763ebb51dfb01b0c9778ca20209308cdcdb"
)
FASTCONV_CALLS = {0: 40, 1: 41, 2: 42, 3: 43}
OUTPUT_FORMATS = {
    0: ((256, 19, 34), "uint8", 5),
    1: ((256, 19, 34), "uint8", 5),
    2: ((256, 19, 34), "uint8", 5),
    3: ((256, 19, 34), "int8", 3),
    4: ((256, 19, 34), "int8", 6),
    5: ((256, 19, 34), "uint8", 4),
    6: ((256, 19, 34), "uint16", -11),
    11: ((256, 19, 34), "uint8", 7),
    13: ((256, 19, 34), "int8", 4),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root", type=Path, default=REPOSITORY / ".runtime/ades-full-kernels"
    )
    parser.add_argument(
        "--oracle-root", type=Path, default=REPOSITORY / ".runtime/ades-split7"
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=REPOSITORY
        / ".runtime/ades-split8-input0/d0-logical-from-full.bin",
    )
    parser.add_argument("--stress-root", type=Path, action="append", default=[])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-split07-mixed-validation.json",
    )
    parser.add_argument(
        "--performance-out",
        type=Path,
        default=REPOSITORY / "evidence/performance/macos-mlx-split07.json",
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _percentile(samples: list[int], percentile: int) -> int:
    ordered = sorted(samples)
    return ordered[max(0, (percentile * len(ordered) + 99) // 100 - 1)]


def _negative_magnitude_float16(input_tensor: np.ndarray) -> np.ndarray:
    """Encode ``-abs(x / 8)`` in signed fp16 with exponent offset -11."""

    scaled = -np.abs(input_tensor.astype(np.int16)) * 256
    return scaled.astype(np.float16).view(np.uint16)


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
            input_channels=251,
        ),
        1: MlxFastconvSession(
            captures[1].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=256,
        ),
        2: MlxFastconvSession(
            captures[2].channels,
            FastconvGeometry(3, 3, 1, 1, 1, 1),
            input_channels=256,
        ),
        3: MlxFastconvSession(
            captures[3].channels,
            FastconvGeometry(1, 1, 1, 1, 0, 0),
            input_channels=256,
            output_dtype="int8",
        ),
    }
    input_bytes = (args.oracle_root / "d0-logical.bin").read_bytes()
    input_tensor = np.frombuffer(input_bytes, dtype=np.uint8).reshape(251, 19, 34)

    def infer() -> dict[int, np.ndarray]:
        output: dict[int, np.ndarray] = {0: sessions[0].infer(input_tensor)}
        output[1] = sessions[1].infer(output[0])
        output[2] = sessions[2].infer(output[1])
        output[3] = sessions[3].infer(output[2])
        output[4] = np.where(output[3] < 0, 64, -64).astype(np.int8)
        output[5] = np.clip(
            np.abs(output[3].astype(np.int16)) * 2, 0, 255
        ).astype(np.uint8)
        output[6] = _negative_magnitude_float16(output[3])
        output[11] = np.where(output[3] < 0, 0, 128).astype(np.uint8)
        output[13] = quantized_silu_int8(output[3])
        return output

    outputs = infer()
    boundaries = []
    for operator_id, (shape, dtype, exponent_offset) in OUTPUT_FORMATS.items():
        actual = outputs[operator_id]
        oracle_bytes = (args.oracle_root / f"op{operator_id}.bin").read_bytes()
        actual_bytes = actual.tobytes(order="C")
        oracle = np.frombuffer(oracle_bytes, dtype=np.dtype(dtype)).reshape(shape)
        passed = (
            actual.shape == shape
            and actual.dtype == np.dtype(dtype)
            and actual_bytes == oracle_bytes
        )
        boundary = {
            "operator_id": operator_id,
            "operator_type": (
                "fastconvolution_operator_t"
                if operator_id in sessions
                else {
                    4: "lvlcurve_operator_t",
                    5: "multiplyadd_operator_t",
                    6: "multiplyadd_operator_t",
                    11: "multiplyadd_operator_t",
                    13: "fused_quantized_silu_output",
                }[operator_id]
            ),
            "shape": list(shape),
            "dtype": dtype,
            "exponent_offset": exponent_offset,
            "size": len(oracle_bytes),
            "oracle_member": f"op{operator_id}.bin",
            "oracle_sha256": _sha256(oracle_bytes),
            "actual_sha256": _sha256(actual_bytes),
            "mismatch_count": int(np.count_nonzero(actual != oracle)),
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
        raise SystemExit("one or more MLX split-7 boundaries differ from ADES")

    stress_roots = [args.oracle_root, *args.stress_root]
    observed: dict[int, int] = {}
    stress_corpus = []
    for root in stress_roots:
        source = np.fromfile(root / "op3.bin", dtype=np.int8)
        expected = np.fromfile(root / "op13.bin", dtype=np.int8)
        actual = quantized_silu_int8(source)
        if not np.array_equal(actual, expected):
            raise SystemExit(f"quantized SiLU differs from ADES stress corpus: {root}")
        for source_value, output_value in zip(source, expected, strict=True):
            previous = observed.setdefault(int(source_value), int(output_value))
            if previous != int(output_value):
                raise SystemExit("ADES SiLU mapping is not deterministic")
        stress_corpus.append(
            {
                "name": root.name,
                "input_sha256": _sha256((root / "op3.bin").read_bytes()),
                "output_sha256": _sha256((root / "op13.bin").read_bytes()),
                "minimum_input": int(source.min()),
                "maximum_input": int(source.max()),
                "unique_input_values": int(np.unique(source).size),
                "mismatch_count": int(np.count_nonzero(actual != expected)),
            }
        )
    domain = sorted(observed)
    contiguous = domain == list(range(domain[0], domain[-1] + 1))

    full_output = args.full_output.read_bytes()
    full_output_passed = outputs[13].tobytes(order="C") == full_output
    if not full_output_passed:
        raise SystemExit("split-7 output differs from full-runtime split-8 input")

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
    if warmed[13].tobytes(order="C") != full_output:
        raise SystemExit("warmed split-7 output differs from full-runtime output")

    validation = {
        "schema": "verkeye.cv22.split-mixed-validation.v1",
        "identity": {
            "model_sha256": MODEL_SHA256,
            "dvi_sha256": DVI_SHA256,
            "split_index": 7,
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
                "descriptor": "d0",
                "member": "d0-logical.bin",
                "size": len(input_bytes),
                "sha256": _sha256(input_bytes),
            }
        ],
        "semantics": {
            "operator_count": 14,
            "families": {
                "fastconvolution_operator_t": 4,
                "lvlcurve_operator_t": 1,
                "multiplyadd_operator_t": 7,
                "transcendental_operator_t": 2,
            },
            "numeric_rules": [
                "four recovered sparse convolutions execute with exact CV22 requantization",
                "signed int8 input uses exponent offset 3",
                "SiLU output uses exponent offset 4",
                "operators 4-13 are fused to 2*x*sigmoid(x/8) with int8 saturation",
                "the fused host mapping is a 256-entry lookup with no runtime transcendental",
            ],
            "fused_internal_operators": [7, 8, 9, 10, 12],
        },
        "boundaries": boundaries,
        "stress_domain": {
            "minimum": domain[0],
            "maximum": domain[-1],
            "observed_values": len(domain),
            "contiguous": contiguous,
            "mapping_sha256": _sha256(
                bytes((observed[value] & 0xFF) for value in domain)
            ),
            "corpus": stress_corpus,
            "comparison": "byte-identical for every observed input value",
        },
        "full_runtime_output_check": {
            "consumer": "split-8 input 0",
            "size": len(full_output),
            "oracle_sha256": _sha256(full_output),
            "actual_sha256": _sha256(outputs[13].tobytes(order="C")),
            "comparison": "byte-identical",
            "passed": full_output_passed,
        },
        "result": {
            "comparison": "byte-identical at all exposed host boundaries",
            "compared_bytes": sum(boundary["size"] for boundary in boundaries),
            "mismatch_count": 0,
            "passed": True,
        },
    }
    performance = {
        "schema": "verkeye.performance-sample.v1",
        "backend": "mlx-cb62-split07-mixed",
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "elapsed_ns": elapsed_ns,
        "fps": args.frames * 1_000_000_000 / elapsed_ns,
        "frame_ns": samples,
        "percentiles_ns": {
            "p50": int(statistics.median(samples)),
            "p95": _percentile(samples, 95),
            "max": max(samples),
        },
        "output_sha256": _sha256(warmed[13].tobytes(order="C")),
        "session_creation_count": sum(
            session.session_creation_count for session in sessions.values()
        ),
        "session_reuse_verified": all(
            session.inference_count == args.warmup + args.frames + 1
            for session in sessions.values()
        ),
    }
    for path, document in (
        (args.validation_out, validation),
        (args.performance_out, performance),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)
    print(json.dumps({"validation": validation["result"], "performance": performance}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
