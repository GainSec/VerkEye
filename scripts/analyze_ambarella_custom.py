#!/usr/bin/env python3
"""Generate the exact bounded CV22 unsupported-instruction inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.compat.ambarella_custom import analyze_ambarella_custom
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    atomic_write_json(args.out, analyze_ambarella_custom(args.firmware))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
