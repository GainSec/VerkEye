#!/usr/bin/env python3
"""Generate the exact OpenRISC evidence report for the CV22 scheduler."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.compat.openrisc import analyze_openrisc_firmware
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    atomic_write_json(args.out, analyze_openrisc_firmware(args.firmware))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
