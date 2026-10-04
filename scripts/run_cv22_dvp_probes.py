#!/usr/bin/env python3
"""Generate bounded positive and negative evidence for CV22 DVP operations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from verkeye.compat.cv22_dvp_probe import (
    analyze_cv22_dvp_evidence,
    build_cv22_dvp_probe,
    build_cv22_dvp_probe_command,
    build_cv22_dvp_unknown_read_probe,
    run_bounded_qemu_probe,
)


QEMU_COMMIT = "f7ada39edacaa5c26b30e98b94017b0b2ccbcf94"


def _single_instruction_tbs(command: tuple[str, ...]) -> tuple[str, ...]:
    items = list(command)
    index = items.index("-m")
    items[index:index] = ["-accel", "tcg,one-insn-per-tb=on"]
    return tuple(items)


def _replace_cpu(command: tuple[str, ...], cpu: str) -> tuple[str, ...]:
    items = list(command)
    items[items.index("-cpu") + 1] = cpu
    return tuple(items)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qemu", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    positive = build_cv22_dvp_probe()
    unknown = build_cv22_dvp_unknown_read_probe()
    positive_image = args.out / "cv22-dvp-write-read-probe.bin"
    positive_trace = args.out / "cv22-dvp-write-read-probe.log"
    noncv_trace = args.out / "cv22-dvp-noncv22-illegal.log"
    unknown_image = args.out / "cv22-dvp-unknown-read-probe.bin"
    unknown_trace = args.out / "cv22-dvp-unknown-read-illegal.log"
    positive_image.write_bytes(positive.image)
    unknown_image.write_bytes(unknown.image)

    positive_command = build_cv22_dvp_probe_command(
        args.qemu, positive_image, positive_trace
    )
    run_bounded_qemu_probe(
        positive_command,
        positive_trace,
        lambda raw: b"PC=0040001c" in raw and b"R06=a5afe5e7" in raw,
        timeout_seconds=args.timeout,
    )

    noncv_command = _replace_cpu(
        build_cv22_dvp_probe_command(args.qemu, positive_image, noncv_trace), "any"
    )
    run_bounded_qemu_probe(
        _single_instruction_tbs(noncv_command),
        noncv_trace,
        lambda raw: b"0x00400010:" in raw and b"PC=00000700" in raw,
        timeout_seconds=args.timeout,
    )

    unknown_command = build_cv22_dvp_probe_command(
        args.qemu, unknown_image, unknown_trace
    )
    run_bounded_qemu_probe(
        _single_instruction_tbs(unknown_command),
        unknown_trace,
        lambda raw: b"0x00400008:" in raw and b"PC=00000700" in raw,
        timeout_seconds=args.timeout,
    )

    report = analyze_cv22_dvp_evidence(
        positive_image=positive_image,
        positive_trace=positive_trace,
        noncv_trace=noncv_trace,
        unknown_image=unknown_image,
        unknown_trace=unknown_trace,
        qemu_commit=QEMU_COMMIT,
        patch_sha256=hashlib.sha256(args.patch.read_bytes()).hexdigest(),
    )
    destination = args.out / "cv22-dvp-evidence.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
