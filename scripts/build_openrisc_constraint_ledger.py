#!/usr/bin/env python3
"""Build the exact CV22 custom-instruction constraint ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verkeye.compat.openrisc_constraints import build_openrisc_constraint_ledger
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--dvp-evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.surface, args.contexts, args.dvp_evidence)
    ]
    atomic_write_json(args.out, build_openrisc_constraint_ledger(*documents))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
