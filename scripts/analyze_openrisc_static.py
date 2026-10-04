#!/usr/bin/env python3
"""Correlate a pinned CV22 firmware disassembly with dynamic boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verkeye.compat.openrisc_static import analyze_openrisc_static
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--disassembly", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--runtime-base", type=lambda value: int(value, 0), default=0x400000)
    parser.add_argument("--context-instructions", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))
    report = analyze_openrisc_static(
        args.firmware,
        args.disassembly,
        boundaries,
        runtime_base=args.runtime_base,
        context_instructions=args.context_instructions,
    )
    atomic_write_json(args.out, report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
