"""Atomic artifact output and deterministic execution manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from . import __version__


ExecutionStatus = Literal["passed", "blocked", "failed"]


def _file_record(source: Path, *, path: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path,
        "size": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def artifact_record(path: str | Path) -> dict[str, Any]:
    """Hash a file or a deterministic recursive directory snapshot."""

    source = Path(path)
    if not source.is_dir():
        return {
            "kind": "file",
            **_file_record(source, path=str(source.resolve())),
        }

    files = [
        _file_record(item, path=item.relative_to(source).as_posix())
        for item in sorted(source.rglob("*"))
        if item.is_file()
    ]
    tree_digest = hashlib.sha256()
    for item in files:
        tree_digest.update(item["path"].encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(item["size"]).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(item["sha256"].encode("ascii"))
        tree_digest.update(b"\n")
    return {
        "kind": "directory",
        "path": str(source.resolve()),
        "size": sum(item["size"] for item in files),
        "file_count": len(files),
        "sha256": tree_digest.hexdigest(),
        "files": files,
    }


def atomic_write_text(path: str | Path, content: str) -> None:
    """Replace a text file only after the complete new content reaches disk."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, document: Mapping[str, Any]) -> None:
    content = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ": "),
    ) + "\n"
    atomic_write_text(path, content)


def execution_manifest(
    *,
    command: str,
    argv: Sequence[str],
    status: ExecutionStatus,
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    gates: Mapping[str, str],
    warnings: Sequence[str],
    incomplete_dispositions: Mapping[str, int],
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a deterministic record of one CLI decision and its artifacts."""

    return {
        "schema": "verkeye.execution-manifest.v1",
        "tool": {"name": "verkeye", "version": __version__},
        "command": {"name": command, "argv": list(argv)},
        "status": status,
        "inputs": [dict(item) for item in inputs],
        "outputs": [dict(item) for item in outputs],
        "gates": dict(gates),
        "warnings": list(warnings),
        "incomplete_dispositions": dict(incomplete_dispositions),
        "blockers": list(blockers),
    }
