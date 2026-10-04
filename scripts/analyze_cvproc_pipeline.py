#!/usr/bin/env python3
"""Generate the pinned CB62 cvproc pipeline evidence report."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.cv22.pipeline import analyze_cvproc_pipeline
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvproc", type=Path, required=True)
    parser.add_argument("--libvproc", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor-map", type=Path, required=True)
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--vconfig", type=Path, required=True)
    parser.add_argument("--bruce-4k-config", type=Path, required=True)
    parser.add_argument("--bruce-4k-telephoto-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    atomic_write_json(
        args.out,
        analyze_cvproc_pipeline(
            args.cvproc,
            args.libvproc,
            args.model,
            args.tensor_map,
            args.production_config,
            args.vconfig,
            args.bruce_4k_config,
            args.bruce_4k_telephoto_config,
        ),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
