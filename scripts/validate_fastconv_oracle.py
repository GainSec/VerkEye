#!/usr/bin/env python3
"""Validate the recovered first CB62 convolution against exact ADES output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from verkeye.cv22.operators import FastconvGeometry, execute_sparse_fastconv_region
from verkeye.cv22.parameter_extraction import (
    encode_fastconv_capture,
    parse_fastconv_capture,
)


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
    parser.add_argument(
        "--fixture",
        type=Path,
        default=(
            REPOSITORY / "fixtures/operators/cb62-split00-op00-channel01.json"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-fastconv-validation.json",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    entries_path = args.capture_root / "fastconv-0-entries.bin"
    entries = entries_path.read_bytes()
    channel_count = len(entries) // 96
    payloads = {
        channel_index: (
            args.capture_root
            / f"fastconv-0-entry-{channel_index}-offset-0.bin"
        ).read_bytes()
        for channel_index in range(channel_count)
    }
    capture = parse_fastconv_capture(entries, payloads)
    input_bytes = args.input.read_bytes()
    input_shape = tuple(fixture["input"]["shape"])
    input_tensor = np.frombuffer(input_bytes, dtype=np.uint8).reshape(input_shape)
    oracle_bytes = args.oracle_output.read_bytes()
    oracle_shape = (len(capture.channels), 304, 544)
    oracle = np.frombuffer(oracle_bytes, dtype=np.uint8).reshape(oracle_shape)
    actual = execute_sparse_fastconv_region(
        input_tensor,
        capture.channels,
        FastconvGeometry(**fixture["geometry"]),
        output_origin=(0, 0),
        output_shape=oracle_shape[1:],
    )
    actual_bytes = actual.tobytes(order="C")
    mismatches = int(np.count_nonzero(actual != oracle))
    document = {
        "schema": "verkeye.cv22.fastconv-validation.v1",
        "identity": fixture["identity"] | {"output_channel": "all"},
        "input": {
            "member": args.input.name,
            "shape": list(input_shape),
            "dtype": "uint8",
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
        },
        "parameters": {
            "entries_member": entries_path.name,
            "entries_sha256": capture.entries_sha256,
            "channel_count": len(capture.channels),
            "point_count": sum(len(channel.points) for channel in capture.channels),
            "normalized_sha256": hashlib.sha256(
                encode_fastconv_capture(capture)
            ).hexdigest(),
        },
        "geometry": fixture["geometry"],
        "numeric_semantics": [
            "signed sparse integer accumulation",
            "arithmetic per-channel accumulator shift",
            "upper saturation at 511 after accumulator shift",
            "signed per-channel offset",
            "one-bit rounded right shift for final_shift_control=2",
            "uint8 saturation",
        ],
        "oracle": {
            "runtime": "digest-pinned ADES 2.4.2",
            "member": args.oracle_output.name,
            "shape": list(oracle_shape),
            "dtype": "uint8",
            "size": len(oracle_bytes),
            "sha256": hashlib.sha256(oracle_bytes).hexdigest(),
        },
        "result": {
            "comparison": "byte-identical",
            "actual_size": len(actual_bytes),
            "actual_sha256": hashlib.sha256(actual_bytes).hexdigest(),
            "mismatch_count": mismatches,
            "passed": mismatches == 0 and actual_bytes == oracle_bytes,
        },
    }
    if document["input"]["sha256"] != fixture["input"]["sha256"]:
        raise SystemExit("input tensor identity differs from fixture")
    if document["oracle"]["sha256"] != fixture["provenance"]["oracle_output_sha256"]:
        raise SystemExit("oracle output identity differs from fixture")
    if not document["result"]["passed"]:
        raise SystemExit(f"fastconv validation failed with {mismatches} mismatches")
    _atomic_json(args.json_out, document)
    print(json.dumps(document["result"], sort_keys=True))
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
