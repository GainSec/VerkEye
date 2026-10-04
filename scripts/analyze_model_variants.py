#!/usr/bin/env python3
"""Emit byte-exact differential evidence for two CV22 model variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verkeye.cv22.differential import inspect_model_variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--minimum-run", type=int, default=64)
    parser.add_argument("--minimum-match-bytes", type=int, default=4096)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    document = inspect_model_variants(
        args.left,
        args.right,
        minimum_run=args.minimum_run,
        minimum_match_bytes=args.minimum_match_bytes,
    )
    serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(serialized, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
