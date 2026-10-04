#!/usr/bin/env python3
"""Build a hashed, explicitly non-faithful OpenRISC reachability probe."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.compat.openrisc_probe import build_openrisc_probe
from verkeye.evidence import atomic_write_json


def _offset(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer offset: {value}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--patch-offset", type=_offset, action="append", required=True)
    parser.add_argument("--image-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()

    probe = build_openrisc_probe(args.firmware, args.patch_offset)
    args.image_out.parent.mkdir(parents=True, exist_ok=True)
    args.image_out.write_bytes(probe.qemu_word_swapped_image)
    atomic_write_json(args.manifest_out, probe.report)
    print(f"wrote {args.image_out}")
    print(f"wrote {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
