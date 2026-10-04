#!/usr/bin/env python3
"""Generate the exact CV22 parameter-recovery gate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from verkeye.cv22.container import parse_container  # noqa: E402
from verkeye.cv22.tensors import parse_tensor_map  # noqa: E402
from verkeye.cv22.weights import (  # noqa: E402
    compare_compiled_packages,
    inspect_weight_map_path,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _model(path: Path) -> tuple[bytes, object]:
    data = path.read_bytes()
    return data, parse_tensor_map(data, parse_container(data))


def _comparison(
    primary_path: Path,
    sibling_path: Path,
    source_corpus: Path | None,
) -> dict[str, object]:
    primary_data, primary_splits = _model(primary_path)
    sibling_data, sibling_splits = _model(sibling_path)
    comparison = compare_compiled_packages(
        primary_data,
        primary_splits,  # type: ignore[arg-type]
        sibling_data,
        sibling_splits,  # type: ignore[arg-type]
    )
    source_path = (
        f"{source_corpus.resolve()}!var/cache/cvproc/{sibling_path.name}"
        if source_corpus is not None
        else str(sibling_path.resolve())
    )
    return {
        "artifact": {
            "path": source_path,
            "size": len(sibling_data),
            "sha256": _sha256(sibling_data),
            "split_count": len(sibling_splits),  # type: ignore[arg-type]
        },
        "method": {
            "chunk_size": comparison.chunk_size,
            "minimum_run_size": comparison.minimum_run_size,
            "anchor_rule": "chunk occurs exactly once in each compiled package",
            "extension_rule": "bytewise maximal match within source split bounds",
        },
        "result": {
            "run_count": len(comparison.runs),
            "shared_primary_bytes": comparison.shared_bytes,
            "runs": [
                {
                    "primary_split": run.primary_split,
                    "primary_relative_offset": run.primary_relative_offset,
                    "secondary_split": run.secondary_split,
                    "secondary_relative_offset": run.secondary_relative_offset,
                    "size": run.size,
                }
                for run in comparison.runs
            ],
        },
        "interpretation_boundary": (
            "a shared run establishes byte stability between compiler outputs; "
            "it does not identify instructions, constants, or parameters"
        ),
    }


def _runtime(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    completed = subprocess.run(
        ["objdump", "-T", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    imports = sorted(
        {
            line.split()[-1]
            for line in completed.stdout.splitlines()
            if " nnctrl_" in line or " cavalry_mem_" in line
        }
    )
    return {
        "path": str(path.resolve()),
        "size": len(data),
        "sha256": _sha256(data),
        "dynamic_imports": imports,
        "conclusion": (
            "the recovered host executable delegates model initialization, "
            "package loading, DVI queries, and execution to proprietary NNCtrl "
            "and Cavalry libraries"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--comparison", action="append", default=[], type=Path)
    parser.add_argument("--source-corpus", type=Path)
    parser.add_argument("--runtime-binary", type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    result = inspect_weight_map_path(args.model)
    result["cross_build_evidence"] = [
        _comparison(args.model, sibling, args.source_corpus)
        for sibling in args.comparison
    ]
    if args.runtime_binary is not None:
        result["runtime_decoder_evidence"] = _runtime(args.runtime_binary)
    result["gate_decision"] = {
        "onnx_export": "stopped",
        "reason": (
            "the weight and quantization gates are blocked; exporting an ONNX "
            "graph would require guessed parameters or numeric semantics"
        ),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
