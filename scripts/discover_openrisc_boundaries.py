#!/usr/bin/env python3
"""Run bounded, explicitly non-faithful CV22 OpenRISC boundary discovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.compat.openrisc_discovery import (
    discover_openrisc_boundaries,
    run_qemu_discovery_iteration,
)
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--qemu", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def provider(iteration: int, image: bytes):
        return run_qemu_discovery_iteration(
            args.qemu,
            args.trace_dir,
            timeout_seconds=args.timeout_seconds,
            iteration=iteration,
            image=image,
        )

    report = discover_openrisc_boundaries(
        args.firmware,
        provider,
        max_iterations=args.max_iterations,
    )
    atomic_write_json(args.out, report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
