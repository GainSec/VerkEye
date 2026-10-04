#!/usr/bin/env python3
"""Generate the bounded CV22 no-delay QEMU compatibility report."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.compat.qemu_cv22 import analyze_cv22_qemu
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    atomic_write_json(
        args.out,
        analyze_cv22_qemu(args.firmware, args.trace, args.patch),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
