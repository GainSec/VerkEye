#!/usr/bin/env python3
"""Generate the deterministic ABI report for the pinned Cavalry driver."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.compat.driver_abi import analyze_cavalry_driver
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    atomic_write_json(args.out, analyze_cavalry_driver(args.driver))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
