#!/usr/bin/env python3
"""Generate exact tensor-descriptor and split-graph evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.cv22.graph import inspect_tensor_graph_path
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    atomic_write_json(args.json_out, inspect_tensor_graph_path(args.model))
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
