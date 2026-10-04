"""Pinned source-to-binary evidence for the recovered Ambarella runtime."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from verkeye.cv22.vendor_runtime import analyze_elf

from .public_cavalry_abi import PINNED_PUBLIC_CAVALRY_SOURCE


class VendorSourceError(ValueError):
    """A source snapshot or recovered binary contradicts the pinned evidence."""


_SNAPSHOT_SCHEMA = "verkeye.vendor-source-snapshot.v1"
_SOURCE_SPECS = {
    "nnctrl": {
        "version": "0.3.0",
        "commit": "aa3856db0028691b688b122a11279d87efd25f65",
        "tree": "b85444bc9bc92d35cbd28e693075e3287670699e",
        "version_header": "src/nnctrl_ver.h",
        "version_prefix": "NNCTRL_LIB",
        "api_header": "inc/nnctrl.h",
    },
    "cavalry_mem": {
        "version": "0.0.6",
        "commit": "1258216ac5ab41285c67a81dee5ef01bd73d0350",
        "tree": "858ed480f82dbc3f3e1cf79d7671ce35a24dadc2",
        "version_header": "src/mem_ver.h",
        "version_prefix": "MEM_LIB",
        "api_header": "inc/cavalry_mem.h",
    },
}
_BINARY_SPECS = {
    "libnnctrl": {
        "source": "nnctrl",
        "size": 132_968,
        "sha256": "647b38c0ef77558bc58bc2b32e1003bd8c4c781eaf25e195268297edc44f4c3d",
    },
    "libcavalry_mem": {
        "source": "cavalry_mem",
        "size": 67_392,
        "sha256": "330836af26ab536d02430ea8ac338fa2975688d290c3c12b0304d4554aac0ed4",
    },
}
_VERSION_DEFINE = re.compile(
    r"^#define\s+(?P<name>[A-Z0-9_]+)\s+(?P<value>\d+)\s*$", re.MULTILINE
)
_API_DECLARATION = re.compile(
    r"\bAMBA_API\s+[A-Za-z_][A-Za-z0-9_\s*]*?\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot_digest(root: Path) -> tuple[int, str]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SOURCE.json"
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return len(files), digest.hexdigest()


def verify_source_snapshot(path: str | Path) -> dict[str, Any]:
    """Verify one vendored source tree against its content-addressed manifest."""

    root = Path(path).resolve()
    manifest_path = root / "SOURCE.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VendorSourceError(f"cannot read source manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema") != _SNAPSHOT_SCHEMA:
        raise VendorSourceError("unsupported vendor source snapshot schema")
    name = manifest.get("name")
    spec = _SOURCE_SPECS.get(name)
    if spec is None:
        raise VendorSourceError(f"unsupported vendor source snapshot {name!r}")
    for field, expected in (
        ("version", spec["version"]),
        ("commit", spec["commit"]),
        ("git_tree", spec["tree"]),
        ("license", "Apache-2.0"),
    ):
        if manifest.get(field) != expected:
            raise VendorSourceError(
                f"source manifest {field} mismatch: expected {expected}, "
                f"observed {manifest.get(field)!r}"
            )

    file_count, content_digest = _snapshot_digest(root)
    if file_count != manifest.get("file_count"):
        raise VendorSourceError(
            "missing source file(s) or unexpected source file(s): "
            f"expected {manifest.get('file_count')}, observed {file_count}"
        )
    if content_digest != manifest.get("content_sha256"):
        raise VendorSourceError(
            "source content sha256 mismatch: "
            f"expected {manifest.get('content_sha256')}, observed {content_digest}"
        )

    version_path = root / str(spec["version_header"])
    try:
        version_text = version_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise VendorSourceError(f"cannot read version header {version_path}: {exc}") from exc
    definitions = {
        match.group("name"): int(match.group("value"))
        for match in _VERSION_DEFINE.finditer(version_text)
    }
    prefix = str(spec["version_prefix"])
    try:
        observed_version = ".".join(
            str(definitions[f"{prefix}_{component}"])
            for component in ("MAJOR", "MINOR", "PATCH")
        )
    except KeyError as exc:
        raise VendorSourceError(f"version header lacks {exc.args[0]}") from exc
    if observed_version != spec["version"]:
        raise VendorSourceError(
            f"source version mismatch: expected {spec['version']}, observed {observed_version}"
        )

    api_path = root / str(spec["api_header"])
    try:
        api_text = api_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise VendorSourceError(f"cannot read API header {api_path}: {exc}") from exc
    declarations = sorted(
        {match.group("name") for match in _API_DECLARATION.finditer(_COMMENT.sub("", api_text))}
    )
    if not declarations:
        raise VendorSourceError(f"source API header {api_path} has no AMBA_API declarations")

    return {
        "name": name,
        "version": observed_version,
        "upstream": manifest["upstream"],
        "commit": manifest["commit"],
        "git_tree": manifest["git_tree"],
        "license": manifest["license"],
        "file_count": file_count,
        "content_sha256": content_digest,
        "api_declarations": declarations,
    }


def _analyze_library(path: Path, role: str, source: dict[str, Any]) -> dict[str, Any]:
    spec = _BINARY_SPECS[role]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise VendorSourceError(f"cannot read recovered library {path}: {exc}") from exc
    digest = _sha256(raw)
    if len(raw) != spec["size"] or digest != spec["sha256"]:
        raise VendorSourceError(
            f"{role} artifact mismatch: expected size {spec['size']} sha256 "
            f"{spec['sha256']}, observed size {len(raw)} sha256 {digest}"
        )
    elf = analyze_elf(path)
    exports = sorted({item["name"] for item in elf["exported_symbols"]})
    declarations = source["api_declarations"]
    missing = sorted(set(exports) - set(declarations))
    extra = sorted(set(declarations) - set(exports))
    if missing or extra:
        raise VendorSourceError(
            f"{role} source/export mismatch: missing {missing}, extra {extra}"
        )
    return {
        "size": len(raw),
        "sha256": digest,
        "source_name": source["name"],
        "source_commit": source["commit"],
        "export_count": len(exports),
        "exports": exports,
        "missing_in_source": missing,
        "extra_in_source": extra,
    }


def _verify_public_abi_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VendorSourceError(f"cannot read public Cavalry ABI report {path}: {exc}") from exc
    if report.get("schema") != "verkeye.compat.public-cavalry-abi.v1":
        raise VendorSourceError("unsupported public Cavalry ABI report schema")
    source = report.get("source")
    expected = PINNED_PUBLIC_CAVALRY_SOURCE
    if not isinstance(source, dict):
        raise VendorSourceError("public Cavalry ABI report lacks source identity")
    for field, value in (
        ("repository", expected.repository),
        ("commit", expected.commit),
        ("ioctl_sha256", expected.ioctl_sha256),
        ("gen_sha256", expected.gen_sha256),
    ):
        if source.get(field) != value:
            raise VendorSourceError(
                f"public Cavalry ABI source {field} mismatch: expected {value}, "
                f"observed {source.get(field)!r}"
            )
    gate = report.get("fidelity_gate")
    if not isinstance(gate, dict) or gate.get("status") != "blocked":
        raise VendorSourceError("public Cavalry ABI report overstates the fidelity gate")
    driver = report.get("driver_numeric_correlation")
    userspace = report.get("userspace_callsite_correlation")
    if not isinstance(driver, dict) or not isinstance(userspace, dict):
        raise VendorSourceError("public Cavalry ABI report lacks correlation evidence")
    return {
        "repository": source["repository"],
        "commit": source["commit"],
        "matching_driver_request_count": driver.get("matching_request_count"),
        "matching_userspace_request_count": userspace.get("numeric_match_count"),
    }


def analyze_vendor_sources(
    *,
    nnctrl_source: str | Path,
    cavalry_mem_source: str | Path,
    nnctrl_binary: str | Path,
    cavalry_mem_binary: str | Path,
    public_abi_report: str | Path,
) -> dict[str, Any]:
    """Prove exact version/API correspondence without claiming a byte-identical build."""

    nnctrl = verify_source_snapshot(nnctrl_source)
    memory = verify_source_snapshot(cavalry_mem_source)
    libraries = {
        "libnnctrl": _analyze_library(Path(nnctrl_binary).resolve(), "libnnctrl", nnctrl),
        "libcavalry_mem": _analyze_library(
            Path(cavalry_mem_binary).resolve(), "libcavalry_mem", memory
        ),
    }
    public_abi = _verify_public_abi_report(Path(public_abi_report).resolve())
    return {
        "schema": "verkeye.compat.vendor-source.v1",
        "status": "exact_version_and_api_surface_match",
        "claim_boundary": (
            "version headers and complete dynamic export sets match; compiler flags "
            "and byte-identical source-to-binary reproduction remain unproven"
        ),
        "sources": {"nnctrl": nnctrl, "cavalry_mem": memory},
        "libraries": libraries,
        "build_gate": {
            "status": "blocked",
            "contemporary_public_headers": {
                "status": "available_as_corroborative_evidence",
                **public_abi,
                "evidence": "evidence/compatibility/public-cavalry-abi.json",
            },
            "exact_cb62_sdk_headers": {
                "status": "unavailable",
                "required": ["cavalry_gen.h", "cavalry_ioctl.h"],
            },
            "reason": (
                "the exact library sources are present and contemporary public headers "
                "corroborate the numeric ABI, but the exact CB62-era SDK headers, "
                "byte-identical layouts, and hidden DVP semantics remain unavailable"
            ),
        },
    }
