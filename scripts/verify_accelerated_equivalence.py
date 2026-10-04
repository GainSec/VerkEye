#!/usr/bin/env python3
"""Prove accelerated CB62 terminal and production-detection parity with ADES."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Mapping

import numpy as np

from verkeye.accelerated.cb62 import Cb62MlxSession
from verkeye.accelerated.openvino import OpenVinoCb62Session
from verkeye.compat.ades import assemble_cb62_predictions
from verkeye.compat.oracle import load_oracle_bundle
from verkeye.compat.terminal_catalog import validate_terminal_parity_catalog
from verkeye.evidence import atomic_write_json
from verkeye.runtime.inference import load_cb62_core_profile
from verkeye.runtime.yolov6 import decode_cb62_raw_head_core


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mlx", "openvino"), required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPOSITORY / "fixtures/oracle/terminal-parity-catalog.json",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-full-kernels",
    )
    parser.add_argument(
        "--split4-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-split4",
    )
    parser.add_argument(
        "--split5-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-split5",
    )
    parser.add_argument("--openvino-device", default="CPU")
    parser.add_argument("--inference-threads", type=int, default=6)
    return parser


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _session(args: argparse.Namespace) -> object:
    common = {
        "capture_root": args.capture_root,
        "split4_root": args.split4_root,
        "split5_root": args.split5_root,
    }
    if args.backend == "mlx":
        return Cb62MlxSession(**common)
    return OpenVinoCb62Session(
        **common,
        device=args.openvino_device,
        inference_threads=args.inference_threads,
    )


def _resolve_repository_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY / candidate


def _validate_case(
    session: object,
    case: Mapping[str, object],
    profile: object,
) -> dict[str, object]:
    bundle = load_oracle_bundle(_resolve_repository_path(str(case["bundle"])))
    input_tensor = np.frombuffer(bundle.input_tensor, dtype=np.uint8).reshape(
        3, 608, 1088
    )
    input_sha256 = _sha256_bytes(input_tensor.tobytes(order="C"))
    if input_sha256 != case["input_sha256"]:
        raise RuntimeError(f"{case['case_id']}: registered model input changed")

    started_ns = time.perf_counter_ns()
    outputs = session.infer_outputs(input_tensor)
    inference_ns = time.perf_counter_ns() - started_ns
    tensor_results: dict[str, object] = {}
    for name, expected in case["terminal_tensors"].items():
        tensor = outputs[name]
        actual_sha256 = _sha256_bytes(tensor.tobytes(order="C"))
        if actual_sha256 != expected["sha256"]:
            raise RuntimeError(
                f"{case['case_id']}/{name}: expected {expected['sha256']}, "
                f"got {actual_sha256}"
            )
        tensor_results[name] = {
            "shape": list(tensor.shape),
            "dtype": tensor.dtype.name,
            "sha256": actual_sha256,
            "differing_bytes": 0,
        }

    predictions = assemble_cb62_predictions(outputs)
    prediction_sha256 = _sha256_bytes(predictions.tobytes(order="C"))
    if prediction_sha256 != case["prediction_sha256"]:
        raise RuntimeError(f"{case['case_id']}: assembled prediction bytes differ")
    detections = decode_cb62_raw_head_core(predictions, profile)
    if list(detections) != case["production_detections"]:
        raise RuntimeError(f"{case['case_id']}: production detections differ")
    return {
        "case_id": case["case_id"],
        "provenance_kind": case["provenance"]["kind"],
        "input_sha256": input_sha256,
        "inference_ns": inference_ns,
        "terminal_tensors": tensor_results,
        "prediction_sha256": prediction_sha256,
        "production_detection_count": len(detections),
        "production_detections_exact": True,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.inference_threads <= 0:
        raise ValueError("--inference-threads must be positive")
    catalog_bytes = args.catalog.read_bytes()
    catalog = json.loads(catalog_bytes)
    cases = validate_terminal_parity_catalog(catalog)
    model_path = _resolve_repository_path(catalog["model"]["path"])
    pipeline_path = _resolve_repository_path(catalog["pipeline"]["path"])
    model_sha256 = _sha256_file(model_path)
    pipeline_sha256 = _sha256_file(pipeline_path)
    if model_sha256 != MODEL_SHA256 or model_sha256 != catalog["model"]["sha256"]:
        raise RuntimeError("the exact recovered CB62 model hash is not present")
    if pipeline_sha256 != catalog["pipeline"]["sha256"]:
        raise RuntimeError("the registered production pipeline changed")

    session = _session(args)
    profile = load_cb62_core_profile(pipeline_path)
    started_ns = time.perf_counter_ns()
    results = [_validate_case(session, case, profile) for case in cases]
    elapsed_ns = time.perf_counter_ns() - started_ns
    category_counts = Counter(item["provenance_kind"] for item in results)
    document = {
        "schema": "verkeye.accelerated-equivalence.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "equivalence_boundary": (
            "identical uint8 CHW bytes entering the recovered CB62 model"
        ),
        "source_commit": _git_commit(),
        "command": [sys.executable, *sys.argv],
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "backend": dict(session.identity),
        "oracle": catalog["oracle"],
        "model": {"path": str(model_path), "sha256": model_sha256},
        "pipeline": {"path": str(pipeline_path), "sha256": pipeline_sha256},
        "catalog": {
            "path": str(args.catalog),
            "sha256": _sha256_bytes(catalog_bytes),
        },
        "case_count": len(results),
        "category_counts": dict(sorted(category_counts.items())),
        "terminal_tensor_count": len(results) * 6,
        "terminal_tensor_differing_bytes": 0,
        "prediction_mismatch_count": 0,
        "production_detection_mismatch_count": 0,
        "elapsed_ns": elapsed_ns,
        "cases": results,
    }
    atomic_write_json(args.json_out, document)
    print(
        json.dumps(
            {
                "status": "passed",
                "backend": args.backend,
                "cases": len(results),
                "terminal_tensors": len(results) * 6,
                "differing_bytes": 0,
                "detection_mismatches": 0,
                "json_out": str(args.json_out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
