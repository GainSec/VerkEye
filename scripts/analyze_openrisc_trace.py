#!/usr/bin/env python3
"""Generate a report for the exact first Ambarella OpenRISC QEMU boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.compat.openrisc_trace import analyze_openrisc_trace
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    atomic_write_json(
        args.out,
        analyze_openrisc_trace(args.firmware, args.trace),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
