#!/usr/bin/env python3
"""Recover exact CV22 sigmoid mappings from one ADES/OpenVINO differential run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verkeye.accelerated.openvino import OpenVinoCb62Session
from verkeye.compat.ades import (
    parse_cavalry_verbose,
    parse_execution_result,
    read_execution_tensors,
)
from verkeye.cv22.sigmoid_recovery import (
    merge_sigmoid_float32_mappings,
    recover_sigmoid_float32_mapping,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--split4-root", type=Path, required=True)
    parser.add_argument("--split5-root", type=Path, required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    run = args.workspace / args.run_id
    manifest = parse_cavalry_verbose(
        (args.workspace / "cavalry-verbose.txt").read_text()
    )
    execution = parse_execution_result((run / "stdout.tsv").read_text())
    oracle = read_execution_tensors(execution, run, manifest)
    source = np.fromfile(run / "input.tensor", dtype=np.uint8).reshape(3, 608, 1088)
    session = OpenVinoCb62Session(
        capture_root=args.capture_root,
        split4_root=args.split4_root,
        split5_root=args.split5_root,
        device=args.device,
        diagnostics=True,
    )
    observed = session.infer_diagnostics(source)
    levels = (
        ("high", "output_0", 2),
        ("medium", "output_1", 2),
        ("low", "output_2", 3),
    )
    combined: dict[int, dict[int, int]] = {2: {}, 3: {}}
    per_level: dict[str, dict[int, int]] = {}
    for level, output, exponent in levels:
        recovered = recover_sigmoid_float32_mapping(
            observed[f"{level}_score_logits"],
            oracle[output],
            input_exponent_offset=exponent,
        )
        combined[exponent] = merge_sigmoid_float32_mappings(
            combined[exponent], recovered
        )
        per_level[level] = recovered
    document = {
        "schema": "verkeye.cv22.sigmoid-recovery.v1",
        "run_id": args.run_id,
        "provider": dict(session.identity),
        "mapping_float32_bits": {
            str(exponent): {
                str(key): value for key, value in sorted(mapping.items())
            }
            for exponent, mapping in combined.items()
        },
        "levels": {
            level: {str(key): value for key, value in sorted(mapping.items())}
            for level, mapping in per_level.items()
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "domains": {
                    str(exponent): {
                        "minimum": min(mapping),
                        "maximum": max(mapping),
                        "mapping_count": len(mapping),
                    }
                    for exponent, mapping in combined.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
