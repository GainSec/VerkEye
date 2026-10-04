#!/usr/bin/env python3
"""Build a code-scoped CV22 opcode-family reachability report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verkeye.compat.openrisc_reachability import analyze_openrisc_reachability
from verkeye.evidence import atomic_write_json


def _range(value: str) -> tuple[int, int]:
    try:
        start, end = value.split(":", 1)
        return int(start, 0), int(end, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range must be START:END") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--disassembly", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--runtime-base", type=lambda value: int(value, 0), default=0x400000)
    parser.add_argument("--entry-point", action="append", type=lambda value: int(value, 0))
    parser.add_argument("--code-range", action="append", type=_range, default=[])
    parser.add_argument("--family-opcode", type=lambda value: int(value, 0), default=0x10)
    parser.add_argument("--context-instructions", type=int, default=4)
    parser.add_argument("--omit-instruction-inventory", action="store_true")
    parser.add_argument(
        "--no-delay-slots",
        action="store_true",
        help="use the CV22 CPUCFGR.ND=1 control-flow profile",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))
    report = analyze_openrisc_reachability(
        args.firmware,
        args.disassembly,
        boundaries,
        runtime_base=args.runtime_base,
        entry_points=tuple(args.entry_point or (0x400000,)),
        code_ranges=tuple(args.code_range),
        family_opcode=args.family_opcode,
        context_instructions=args.context_instructions,
        include_instruction_inventory=not args.omit_instruction_inventory,
        delay_slots=not args.no_delay_slots,
    )
    atomic_write_json(args.out, report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
