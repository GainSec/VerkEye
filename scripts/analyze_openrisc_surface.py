#!/usr/bin/env python3
"""Generate the aggregate exact CV22 custom-instruction surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verkeye.compat.openrisc_surface import analyze_openrisc_surface
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--disassembly", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))
    report = analyze_openrisc_surface(
        args.firmware,
        args.disassembly,
        boundaries,
    )
    atomic_write_json(args.out, report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
