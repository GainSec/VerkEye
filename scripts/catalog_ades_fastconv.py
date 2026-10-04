#!/usr/bin/env python3
"""Normalize and catalog exact fast-convolution kernels captured from ADES."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from verkeye.cv22.parameter_extraction import catalog_fastconv_captures


REPOSITORY = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-full-kernels",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-ades-dag-semantics.json",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPOSITORY / "evidence/accelerated/cb62-fastconv-catalog.json",
    )
    parser.add_argument(
        "--normalized-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-normalized-fastconv",
    )
    return parser


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    catalog, normalized = catalog_fastconv_captures(args.capture_root, graph)
    for member_name, payload in normalized.items():
        _atomic_write(args.normalized_root / member_name, payload)
    _atomic_write(
        args.catalog,
        (json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "catalog": str(args.catalog),
                "capture_count": catalog["capture_count"],
                "channel_count": catalog["channel_count"],
                "point_count": catalog["point_count"],
                "normalized_root": str(args.normalized_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
