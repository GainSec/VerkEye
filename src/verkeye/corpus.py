"""Verified manifests for recovered VerkEye artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CorpusError(ValueError):
    """The corpus manifest or one of its artifacts is not trustworthy."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{field} must be a non-empty string")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    role: str
    path: Path
    size: int | None
    sha256: str | None
    architecture: str
    source: str
    required: bool
    available: bool

    @classmethod
    def from_document(cls, document: Any, base: Path) -> "ArtifactSpec":
        if not isinstance(document, dict):
            raise CorpusError("artifact entry must be an object")
        role = _text(document.get("role"), "role")
        raw_path = _text(document.get("path"), f"path for {role}")
        architecture = _text(
            document.get("architecture"), f"architecture for {role}"
        )
        source = _text(document.get("source"), f"source for {role}")
        required = document.get("required")
        available = document.get("available")
        if not isinstance(required, bool):
            raise CorpusError(f"required for {role} must be boolean")
        if not isinstance(available, bool):
            raise CorpusError(f"available for {role} must be boolean")

        raw_size = document.get("size")
        if raw_size is not None and (
            not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size < 0
        ):
            raise CorpusError(f"size for {role} must be a non-negative integer")
        raw_sha256 = document.get("sha256")
        if raw_sha256 is not None and (
            not isinstance(raw_sha256, str)
            or len(raw_sha256) != 64
            or any(character not in "0123456789abcdef" for character in raw_sha256)
        ):
            raise CorpusError(f"sha256 for {role} must be lowercase hexadecimal")
        if available and (raw_size is None or raw_sha256 is None):
            raise CorpusError(
                f"available artifact {role} requires size and sha256"
            )

        source_path = Path(raw_path)
        resolved_path = (
            source_path if source_path.is_absolute() else base / source_path
        ).resolve()
        return cls(
            role=role,
            path=resolved_path,
            size=raw_size,
            sha256=raw_sha256,
            architecture=architecture,
            source=source,
            required=required,
            available=available,
        )


@dataclass(frozen=True, slots=True)
class ArtifactCorpus:
    name: str
    manifest: Path
    artifacts: tuple[ArtifactSpec, ...]

    @classmethod
    def from_path(cls, path: str | Path) -> "ArtifactCorpus":
        manifest = Path(path).resolve()
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusError(f"cannot load corpus manifest {manifest}: {exc}") from exc
        if not isinstance(document, dict):
            raise CorpusError("corpus manifest must be an object")
        if document.get("schema") != "verkeye.artifact-corpus.v1":
            raise CorpusError("unsupported corpus schema")
        name = _text(document.get("name"), "name")
        raw_artifacts = document.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise CorpusError("artifacts must be an array")
        artifacts = tuple(
            ArtifactSpec.from_document(item, manifest.parent)
            for item in raw_artifacts
        )
        roles: set[str] = set()
        for artifact in artifacts:
            if artifact.role in roles:
                raise CorpusError(f"duplicate artifact role: {artifact.role}")
            roles.add(artifact.role)
        return cls(name=name, manifest=manifest, artifacts=artifacts)

    def verify(self) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for artifact in self.artifacts:
            exists = artifact.path.is_file()
            if not artifact.available:
                if exists:
                    raise CorpusError(
                        f"availability mismatch for {artifact.role}: file exists"
                    )
                if artifact.required:
                    raise CorpusError(
                        f"required artifact {artifact.role} is unavailable"
                    )
                status = "unavailable_optional"
            else:
                if not exists:
                    raise CorpusError(f"artifact missing for {artifact.role}")
                actual_size = artifact.path.stat().st_size
                if actual_size != artifact.size:
                    raise CorpusError(
                        f"size mismatch for {artifact.role}: "
                        f"expected {artifact.size}, observed {actual_size}"
                    )
                actual_sha256 = _sha256(artifact.path)
                if actual_sha256 != artifact.sha256:
                    raise CorpusError(
                        f"sha256 mismatch for {artifact.role}: "
                        f"expected {artifact.sha256}, observed {actual_sha256}"
                    )
                status = "verified"
            results.append(
                {
                    "role": artifact.role,
                    "path": str(artifact.path),
                    "size": artifact.size,
                    "sha256": artifact.sha256,
                    "architecture": artifact.architecture,
                    "source": artifact.source,
                    "required": artifact.required,
                    "available": artifact.available,
                    "status": status,
                }
            )
        return {
            "schema": "verkeye.artifact-corpus-report.v1",
            "name": self.name,
            "manifest": str(self.manifest),
            "status": "passed",
            "artifacts": results,
        }

