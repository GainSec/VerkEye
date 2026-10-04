#!/usr/bin/env python3
"""Generate the source-and-binary proof for NNCtrl tensor descriptors."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.cv22.descriptor_abi import analyze_descriptor_abi
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    atomic_write_json(
        args.out,
        analyze_descriptor_abi(args.library, args.source),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
