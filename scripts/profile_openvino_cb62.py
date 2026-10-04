#!/usr/bin/env python3
"""Report the slowest operations in one exact OpenVINO CB62 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verkeye.accelerated.openvino import OpenVinoCb62Session
from verkeye.compat.oracle import load_oracle_bundle


REPOSITORY = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--case", default=".runtime/oracle/sg-camera-2704")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    session = OpenVinoCb62Session(
        capture_root=REPOSITORY / ".runtime/ades-full-kernels",
        split4_root=REPOSITORY / ".runtime/ades-split4",
        split5_root=REPOSITORY / ".runtime/ades-split5",
        device=args.device,
    )
    bundle = load_oracle_bundle(REPOSITORY / args.case)
    tensor = np.frombuffer(bundle.input_tensor, dtype=np.uint8).reshape(3, 608, 1088)
    session.infer_outputs(tensor)
    records = []
    for item in session._request.profiling_info:
        duration_us = int(item.real_time.total_seconds() * 1_000_000)
        records.append(
            {
                "node_name": item.node_name,
                "node_type": item.node_type,
                "exec_type": item.exec_type,
                "status": str(item.status),
                "duration_us": duration_us,
            }
        )
    records.sort(key=lambda item: item["duration_us"], reverse=True)
    print(json.dumps(records[: args.limit], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
