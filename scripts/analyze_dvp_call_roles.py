#!/usr/bin/env python3
"""Build exact named DVP wrapper call-role evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verkeye.compat.dvp_call_roles import analyze_dvp_call_roles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--firmware", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = analyze_dvp_call_roles(args.firmware)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
