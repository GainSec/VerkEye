#!/usr/bin/env python3
"""Generate the CV22 no-delay control-flow correction record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verkeye.compat.openrisc_correction import analyze_control_flow_correction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-context", type=Path, required=True)
    parser.add_argument("--new-context", type=Path, required=True)
    parser.add_argument("--old-discovery", type=Path, required=True)
    parser.add_argument("--new-discovery", type=Path, required=True)
    parser.add_argument("--fallthrough-proof", type=Path, required=True)
    parser.add_argument("--old-patch-sha256", required=True)
    parser.add_argument("--new-patch-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = analyze_control_flow_correction(
        old_context=args.old_context,
        new_context=args.new_context,
        old_discovery=args.old_discovery,
        new_discovery=args.new_discovery,
        fallthrough_proof=args.fallthrough_proof,
        old_patch_sha256=args.old_patch_sha256,
        new_patch_sha256=args.new_patch_sha256,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
