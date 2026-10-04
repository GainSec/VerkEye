#!/usr/bin/env python3
"""Correlate unresolved scheduler words with recovered CV22 ORC blobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verkeye.compat.openrisc_corpus import analyze_openrisc_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    report = analyze_openrisc_corpus(ledger, args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

