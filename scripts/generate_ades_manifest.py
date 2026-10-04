#!/usr/bin/env python3
"""Convert strict ``cavalry_gen -v`` evidence into the native ADES contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from verkeye.compat.ades import parse_cavalry_verbose, render_native_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("verbose", type=Path)
    parser.add_argument("--native-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    manifest = parse_cavalry_verbose(args.verbose.read_text())
    args.native_out.parent.mkdir(parents=True, exist_ok=True)
    args.native_out.write_text(render_native_manifest(manifest))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
