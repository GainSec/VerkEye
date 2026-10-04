#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verkeye.compat.visorc_boot import analyze_visorc_boot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover the exact CB62 VISORC boot/register contract"
    )
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    document = analyze_visorc_boot(args.driver, args.firmware)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
