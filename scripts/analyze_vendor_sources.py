#!/usr/bin/env python3
"""Verify pinned Ambarella sources against the recovered runtime libraries."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.compat.vendor_source import analyze_vendor_sources
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnctrl-source", type=Path, required=True)
    parser.add_argument("--cavalry-mem-source", type=Path, required=True)
    parser.add_argument("--nnctrl-binary", type=Path, required=True)
    parser.add_argument("--cavalry-mem-binary", type=Path, required=True)
    parser.add_argument("--public-abi-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(
        args.out,
        analyze_vendor_sources(
            nnctrl_source=args.nnctrl_source,
            cavalry_mem_source=args.cavalry_mem_source,
            nnctrl_binary=args.nnctrl_binary,
            cavalry_mem_binary=args.cavalry_mem_binary,
            public_abi_report=args.public_abi_report,
        ),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
