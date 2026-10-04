#!/usr/bin/env python3
"""Correlate pinned public Cavalry headers with the recovered CB62 ABI."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.compat.public_cavalry_abi import (
    PINNED_PUBLIC_CAVALRY_SOURCE,
    analyze_public_cavalry_abi,
    load_public_cavalry_source,
)
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ioctl-header", type=Path, required=True)
    parser.add_argument("--gen-header", type=Path, required=True)
    parser.add_argument("--driver-report", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = (
        load_public_cavalry_source(args.source_metadata)
        if args.source_metadata is not None
        else PINNED_PUBLIC_CAVALRY_SOURCE
    )
    atomic_write_json(
        args.out,
        analyze_public_cavalry_abi(
            ioctl_header=args.ioctl_header,
            gen_header=args.gen_header,
            driver_report=args.driver_report,
            source=source,
        ),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
