#!/usr/bin/env python3
"""Capture fail-closed CPU context for CV22 OpenRISC boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verkeye.compat.openrisc_context import (
    capture_openrisc_contexts,
    run_qemu_cpu_trace_iteration,
)
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--qemu", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read discovery report: {exc}")

    def provider(iteration: int, probe: Path):
        return run_qemu_cpu_trace_iteration(
            args.qemu,
            args.trace_dir,
            timeout_seconds=args.timeout_seconds,
            iteration=iteration,
            probe=probe,
        )

    report = capture_openrisc_contexts(discovery, args.probe_dir, provider)
    atomic_write_json(args.out, report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
