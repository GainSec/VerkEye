#!/usr/bin/env python3
"""Generate deterministic reports for all available AArch64 corpus ELFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.corpus import ArtifactCorpus
from verkeye.cv22.vendor_runtime import analyze_elf, supports_static_elf_analysis
from verkeye.evidence import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    corpus = ArtifactCorpus.from_path(args.corpus)
    corpus.verify()
    written = 0
    for artifact in corpus.artifacts:
        if not artifact.available or not supports_static_elf_analysis(
            artifact.architecture
        ):
            continue
        destination = args.out / f"{artifact.role}.json"
        atomic_write_json(destination, analyze_elf(artifact.path))
        print(f"wrote {destination}")
        written += 1
    if written == 0:
        raise SystemExit("no available linux-aarch64 artifacts in corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
