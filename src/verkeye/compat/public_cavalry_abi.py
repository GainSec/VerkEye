"""Correlate contemporary public Cavalry headers with the recovered CB62 ABI.

The public source is useful corroboration, not an exact historical SDK.  This
module deliberately keeps those two claims separate and leaves the execution
fidelity gate closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .cavalry import CAVALRY_REQUESTS


class PublicCavalryAbiError(ValueError):
    """Public source evidence is missing, malformed, or contradicts its pin."""


@dataclass(frozen=True, slots=True)
class PublicCavalrySource:
    repository: str
    commit: str
    ioctl_repository_path: str
    ioctl_sha256: str
    gen_repository_path: str
    gen_sha256: str
    first_header_commit: str
    first_header_date: str


PINNED_PUBLIC_CAVALRY_SOURCE = PublicCavalrySource(
    repository="https://github.com/cchiou-amba/amba-virt",
    commit="71b1b786689f690248524b5139838b0a778047c7",
    ioctl_repository_path="guest-os/linux/amba-cavalry/include/cavalry_ioctl.h",
    ioctl_sha256="9487336bb586410fe36d6d75c55d60304c3829ec834e154eb76beadff22ded0c",
    gen_repository_path="guest-os/linux/amba-cavalry/include/cavalry_gen.h",
    gen_sha256="809853e4e888d3468547979cf9ceb97c737e88e3fb622a1ccede9f0fe25bc965",
    first_header_commit="d7792bc1f891c73418959a63211a63b4307e2303",
    first_header_date="2026-09-15T12:22:31+08:00",
)


def load_public_cavalry_source(path: str | Path) -> PublicCavalrySource:
    """Load an explicit source pin used for deterministic fixture analysis."""

    source_path = Path(path)
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
        source = PublicCavalrySource(**document)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise PublicCavalryAbiError(
            f"cannot read public Cavalry source metadata {source_path}: {exc}"
        ) from exc
    for field in ("commit", "first_header_commit"):
        value = getattr(source, field)
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise PublicCavalryAbiError(f"public source {field} is not a full Git commit")
    for field in ("ioctl_sha256", "gen_sha256"):
        value = getattr(source, field)
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise PublicCavalryAbiError(f"public source {field} is not a SHA-256 digest")
    return source


_IOCTL_DEFINE = re.compile(
    r"^#define\s+(?P<name>CAVALRY_[A-Z0-9_]+)\s+"
    r"_(?P<operation>IOWR|IOR|IOW)\s*"
    r"\(\s*'(?P<type>.)'\s*,\s*(?P<number>0x[0-9A-Fa-f]+)\s*,"
    r"\s*(?P<argument>[^)]+?)\s*\)\s*$",
    re.MULTILINE,
)
_GEN_DEFINE = re.compile(
    r"^#define\s+CAVALRY_GEN_VER_(?P<component>MAJOR|MINOR|PATCH)"
    r"\s+\(?(?P<value>0x[0-9A-Fa-f]+|\d+)\)?\s*$",
    re.MULTILINE,
)
_CV22_ARCH = re.compile(r"\bCAVALRY_VDG_ARCH_CV22\s*=\s*(0x[0-9A-Fa-f]+|\d+)")

_DIRECTION = {"IOW": (1, "write"), "IOR": (2, "read"), "IOWR": (3, "read_write")}
_DESCRIPTOR_SEMANTIC_MARKERS = (
    "dag_loop_cnt",
    "dvi_dag_vaddr",
    "dvi_dram_addr",
    "dvi_img_size",
    "dvi_img_vaddr",
    "poke_bsize",
    "poke_cnt",
    "poke_val",
    "poke_vaddr",
    "port_boffset_in_dag",
    "port_cnt",
    "port_daddr_increment",
    "port_dram_addr",
    "port_dram_size",
    "use_ping_pong_vmem",
)


def _read_verified(path: Path, *, label: str, expected_sha256: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PublicCavalryAbiError(f"cannot read {label} {path}: {exc}") from exc
    observed = hashlib.sha256(data).hexdigest()
    if observed != expected_sha256:
        raise PublicCavalryAbiError(
            f"{label} sha256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return data


def _ioctl_request(operation: str, type_character: str, number: int) -> tuple[int, str]:
    direction_number, direction = _DIRECTION[operation]
    pointer_size = 8
    return (
        (direction_number << 30)
        | (pointer_size << 16)
        | (ord(type_character) << 8)
        | number,
        direction,
    )


def _parse_pointer_abi(text: str) -> list[dict[str, object]]:
    definitions: dict[str, dict[str, object]] = {}
    for match in _IOCTL_DEFINE.finditer(text):
        argument = " ".join(match.group("argument").split())
        if not argument.endswith("*"):
            continue
        name = match.group("name")
        number = int(match.group("number"), 0)
        type_character = match.group("type")
        request, direction = _ioctl_request(
            match.group("operation"), type_character, number
        )
        definition = {
            "name": name,
            "request": request,
            "direction": direction,
            "type_character": type_character,
            "number": number,
            "argument_type": argument,
            "argument_size": 8,
        }
        previous = definitions.get(name)
        if previous is not None and previous != definition:
            raise PublicCavalryAbiError(f"conflicting pointer ABI definitions for {name}")
        definitions[name] = definition
    if not definitions:
        raise PublicCavalryAbiError("public ioctl header has no pointer-sized definitions")
    requests = [int(item["request"]) for item in definitions.values()]
    if len(requests) != len(set(requests)):
        raise PublicCavalryAbiError("public ioctl header assigns one request to multiple names")
    return sorted(definitions.values(), key=lambda item: (int(item["request"]), str(item["name"])))


def _parse_generator_header(text: str) -> dict[str, object]:
    components = {
        match.group("component"): int(match.group("value"), 0)
        for match in _GEN_DEFINE.finditer(text)
    }
    missing = sorted({"MAJOR", "MINOR", "PATCH"} - components.keys())
    if missing:
        raise PublicCavalryAbiError(
            "public generator header lacks version components: " + ", ".join(missing)
        )
    arch = _CV22_ARCH.search(text)
    if arch is None:
        raise PublicCavalryAbiError("public generator header lacks CV22 architecture value")
    return {
        "version": ".".join(str(components[name]) for name in ("MAJOR", "MINOR", "PATCH")),
        "cv22_architecture_value": int(arch.group(1), 0),
    }


def _load_driver_report(path: Path) -> dict[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicCavalryAbiError(f"cannot read recovered driver report {path}: {exc}") from exc
    if report.get("schema") != "verkeye.compat.cavalry-driver-abi.v1":
        raise PublicCavalryAbiError("unsupported recovered Cavalry driver report schema")
    requests = report.get("requests")
    if not isinstance(requests, list) or not requests:
        raise PublicCavalryAbiError("recovered Cavalry driver report has no requests")
    observed: set[int] = set()
    for item in requests:
        if not isinstance(item, dict) or not isinstance(item.get("request"), int):
            raise PublicCavalryAbiError("recovered driver request record is malformed")
        request = int(item["request"])
        if request in observed:
            raise PublicCavalryAbiError(
                f"recovered driver report repeats request 0x{request:08x}"
            )
        observed.add(request)
    return report


def _default_userspace_requests() -> dict[int, str]:
    return {request: spec.name for request, spec in CAVALRY_REQUESTS.items()}


def analyze_public_cavalry_abi(
    *,
    ioctl_header: str | Path,
    gen_header: str | Path,
    driver_report: str | Path,
    source: PublicCavalrySource = PINNED_PUBLIC_CAVALRY_SOURCE,
    recovered_userspace_requests: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Create a claim-bounded correlation report from pinned public headers."""

    ioctl_data = _read_verified(
        Path(ioctl_header), label="ioctl header", expected_sha256=source.ioctl_sha256
    )
    gen_data = _read_verified(
        Path(gen_header), label="generator header", expected_sha256=source.gen_sha256
    )
    try:
        ioctl_text = ioctl_data.decode("utf-8")
        gen_text = gen_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicCavalryAbiError(f"public header is not UTF-8: {exc}") from exc

    public_definitions = _parse_pointer_abi(ioctl_text)
    public_by_request = {int(item["request"]): item for item in public_definitions}
    recovered = _load_driver_report(Path(driver_report))
    driver_by_request = {
        int(item["request"]): item for item in recovered["requests"]  # type: ignore[index]
    }
    common = sorted(public_by_request.keys() & driver_by_request.keys())
    recovered_only = sorted(driver_by_request.keys() - public_by_request.keys())
    public_only = sorted(public_by_request.keys() - driver_by_request.keys())

    userspace = dict(
        _default_userspace_requests()
        if recovered_userspace_requests is None
        else recovered_userspace_requests
    )
    userspace_common = sorted(userspace.keys() & public_by_request.keys())
    exact_names = [
        request
        for request in userspace_common
        if userspace[request] == public_by_request[request]["name"]
    ]
    name_drift = [
        {
            "request": request,
            "recovered_name": userspace[request],
            "public_name": public_by_request[request]["name"],
        }
        for request in userspace_common
        if userspace[request] != public_by_request[request]["name"]
    ]

    source_document = asdict(source)
    source_document.pop("first_header_commit")
    source_document.pop("first_header_date")
    source_document["headers_first_appeared"] = {
        "commit": source.first_header_commit,
        "date": source.first_header_date,
    }
    return {
        "schema": "verkeye.compat.public-cavalry-abi.v1",
        "source": source_document,
        "generator_header": _parse_generator_header(gen_text),
        "public_ioctl_abi": {
            "type_character": "C",
            "argument_size": 8,
            "definition_count": len(public_definitions),
            "definitions": public_definitions,
        },
        "recovered_driver": {
            "artifact": recovered.get("artifact"),
            "request_count": len(driver_by_request),
        },
        "driver_numeric_correlation": {
            "matching_request_count": len(common),
            "matching": [
                {
                    "request": request,
                    "public_name": public_by_request[request]["name"],
                    "recovered_handler": driver_by_request[request].get("handler"),
                }
                for request in common
            ],
            "recovered_only": [
                {
                    "request": request,
                    "recovered_handler": driver_by_request[request].get("handler"),
                }
                for request in recovered_only
            ],
            "public_only": [
                {"request": request, "public_name": public_by_request[request]["name"]}
                for request in public_only
            ],
        },
        "userspace_callsite_correlation": {
            "recovered_request_count": len(userspace),
            "numeric_match_count": len(userspace_common),
            "missing_from_public": [
                {"request": request, "recovered_name": userspace[request]}
                for request in sorted(userspace.keys() - public_by_request.keys())
            ],
            "exact_name_match_count": len(exact_names),
            "exact_name_matches": exact_names,
            "name_drift": name_drift,
        },
        "descriptor_semantic_markers": {
            "present": [
                marker
                for marker in _DESCRIPTOR_SEMANTIC_MARKERS
                if re.search(rf"\b{re.escape(marker)}\b", ioctl_text)
            ],
            "claim": (
                "names corroborate field roles also used by the recovered NNCtrl "
                "source; offsets and structure sizes are not inherited from this header"
            ),
        },
        "fidelity_gate": {
            "status": "blocked",
            "contemporary_header_status": "available_as_corroborative_evidence",
            "exact_cb62_header_status": "unavailable",
            "byte_exact_layout_status": "not_proven",
            "hidden_dvp_instruction_semantics_status": "not_present",
            "reason": (
                "the public headers first appeared in 2026 and their structures have "
                "version drift from the recovered CB62 ABI; they corroborate request "
                "numbers and field roles but cannot supply byte-exact CB62 layouts or "
                "CV22 DVP execution semantics"
            ),
        },
    }
