#!/usr/bin/env python3
"""Benchmark the persistent MLX lowering of the first exact CB62 convolution."""

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
from verkeye.cv22.operators import FastconvGeometry
from verkeye.cv22.parameter_extraction import parse_fastconv_capture


REPOSITORY = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root", type=Path, default=REPOSITORY / ".runtime/ades-direct"
    )
    parser.add_argument(
        "--input", type=Path, default=REPOSITORY / ".runtime/ades-direct/d12.bin"
    )
    parser.add_argument(
        "--oracle-output",
        type=Path,
        default=REPOSITORY / ".runtime/ades-direct/op0.bin",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPOSITORY / "evidence/performance/macos-mlx-fastconv.json",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 1 or args.frames < 1:
        raise SystemExit("warmup and frames must be positive")
    entries = (args.capture_root / "fastconv-0-entries.bin").read_bytes()
    channel_count = len(entries) // 96
    capture = parse_fastconv_capture(
        entries,
        {
            index: (
                args.capture_root / f"fastconv-0-entry-{index}-offset-0.bin"
            ).read_bytes()
            for index in range(channel_count)
        },
    )
    input_bytes = args.input.read_bytes()
    input_tensor = np.frombuffer(input_bytes, dtype=np.uint8).reshape(3, 608, 1088)
    oracle_bytes = args.oracle_output.read_bytes()
    session = MlxFastconvSession(
        capture.channels,
        FastconvGeometry(3, 3, 2, 2, 1, 1),
        input_channels=3,
    )

    for _ in range(args.warmup):
        session.infer(input_tensor)
    samples: list[int] = []
    actual = None
    started = time.perf_counter_ns()
    for _ in range(args.frames):
        frame_started = time.perf_counter_ns()
        actual = session.infer(input_tensor)
        samples.append(time.perf_counter_ns() - frame_started)
    elapsed_ns = time.perf_counter_ns() - started
    assert actual is not None
    actual_bytes = actual.tobytes(order="C")
    if actual_bytes != oracle_bytes:
        raise SystemExit("MLX output differs from exact ADES oracle")
    ordered = sorted(samples)
    document = {
        "schema": "verkeye.performance.macos-mlx-fastconv.v1",
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "provider": dict(session.identity),
        "scope": {
            "model_sha256": "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf",
            "split_index": 0,
            "operator_id": 0,
            "input_shape": [3, 608, 1088],
            "output_shape": [22, 304, 544],
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "output_sha256": hashlib.sha256(actual_bytes).hexdigest(),
            "oracle": "digest-pinned ADES 2.4.2",
            "comparison": "byte-identical",
        },
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "elapsed_ns": elapsed_ns,
        "fps": args.frames * 1_000_000_000 / elapsed_ns,
        "latency_ns": {
            "raw": samples,
            "p50": int(statistics.median(ordered)),
            "p95": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
            "max": max(ordered),
        },
        "session_creation_count": session.session_creation_count,
        "inference_count": session.inference_count,
    }
    _atomic_json(args.json_out, document)
    print(json.dumps({"fps": document["fps"], "latency_ns": document["latency_ns"]}))
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
