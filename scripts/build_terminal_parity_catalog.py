#!/usr/bin/env python3
"""Build the compact six-terminal parity catalog from ADES oracle bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verkeye.compat.ades import parse_cavalry_verbose
from verkeye.compat.terminal_catalog import (
    build_terminal_parity_catalog,
    canonical_json_bytes,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-catalog", type=Path, required=True)
    parser.add_argument("--cavalry-verbose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.oracle_catalog.read_text(encoding="utf-8"))
    manifest = parse_cavalry_verbose(
        args.cavalry_verbose.read_text(encoding="utf-8")
    )
    document = build_terminal_parity_catalog(source, manifest=manifest)
    payload = canonical_json_bytes(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "case_count": document["case_count"],
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
