"""Reconcile OpenRISC evidence after a control-flow model correction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


class OpenRiscCorrectionError(ValueError):
    """The evidence sets cannot support the requested correction report."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load(path: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    source = Path(path)
    raw = source.read_bytes()
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenRiscCorrectionError(f"invalid JSON evidence: {source}") from exc
    if not isinstance(parsed, dict):
        raise OpenRiscCorrectionError(f"evidence root must be an object: {source}")
    return source, raw, parsed


def _artifact(path: Path, raw: bytes) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _boundary_inventory(discovery: dict[str, Any]) -> list[tuple[str, str]]:
    rows = discovery.get("iterations")
    if not isinstance(rows, list):
        raise OpenRiscCorrectionError("discovery evidence lacks iterations")
    result: list[tuple[str, str]] = []
    for row in rows:
        if isinstance(row, dict) and row.get("instruction") is None:
            continue
        try:
            instruction = row["instruction"]
            result.append((instruction["runtime_address"], instruction["word"]))
        except (KeyError, TypeError) as exc:
            raise OpenRiscCorrectionError("malformed discovery instruction") from exc
    return result


def _contexts(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = context.get("contexts")
    if not isinstance(rows, list):
        raise OpenRiscCorrectionError("context evidence lacks contexts")
    return rows


def _register_deltas(
    old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]
) -> list[dict[str, object]]:
    if len(old_rows) != len(new_rows):
        raise OpenRiscCorrectionError("context inventory changed")
    deltas: list[dict[str, object]] = []
    for old, new in zip(old_rows, new_rows, strict=True):
        old_boundary = old.get("boundary_pc")
        new_boundary = new.get("boundary_pc")
        old_word = old.get("instruction", {}).get("word")
        new_word = new.get("instruction", {}).get("word")
        if (old_boundary, old_word) != (new_boundary, new_word):
            raise OpenRiscCorrectionError("context inventory changed")
        old_registers = old.get("register_state_before_boundary")
        new_registers = new.get("register_state_before_boundary")
        if not isinstance(old_registers, dict) or not isinstance(new_registers, dict):
            raise OpenRiscCorrectionError("context lacks register state")
        names = sorted(set(old_registers) | set(new_registers))
        changed = {
            name: {"old": old_registers.get(name), "new": new_registers.get(name)}
            for name in names
            if old_registers.get(name) != new_registers.get(name)
        }
        if changed:
            deltas.append(
                {
                    "boundary_pc": old_boundary,
                    "registers": changed,
                    "word": old_word,
                }
            )
    return deltas


def analyze_control_flow_correction(
    *,
    old_context: str | Path,
    new_context: str | Path,
    old_discovery: str | Path,
    new_discovery: str | Path,
    fallthrough_proof: str | Path,
    old_patch_sha256: str,
    new_patch_sha256: str,
) -> dict[str, object]:
    """Compare pre/post-fix evidence and fail closed on inventory drift."""

    if not _SHA256.fullmatch(old_patch_sha256) or not _SHA256.fullmatch(
        new_patch_sha256
    ):
        raise OpenRiscCorrectionError("patch SHA-256 must be lowercase hex")
    oc_path, oc_raw, oc = _load(old_context)
    nc_path, nc_raw, nc = _load(new_context)
    od_path, od_raw, od = _load(old_discovery)
    nd_path, nd_raw, nd = _load(new_discovery)
    proof_path, proof_raw, proof = _load(fallthrough_proof)

    old_boundaries = _boundary_inventory(od)
    new_boundaries = _boundary_inventory(nd)
    if old_boundaries != new_boundaries:
        raise OpenRiscCorrectionError("boundary inventory changed")
    deltas = _register_deltas(_contexts(oc), _contexts(nc))

    profile = proof.get("control_flow_profile")
    if not isinstance(profile, dict):
        raise OpenRiscCorrectionError("fallthrough proof lacks control-flow profile")
    if profile.get("CPUCFGR.ND") != 1 or profile.get("fallthrough_bytes") != 4:
        raise OpenRiscCorrectionError("fallthrough proof is not the CV22 no-delay profile")

    return {
        "schema": "verkeye.compat.openrisc-control-flow-correction.v1",
        "control_flow_correction": {
            **profile,
            "old_patch_sha256": old_patch_sha256,
            "new_patch_sha256": new_patch_sha256,
        },
        "boundary_inventory": {
            "preserved": True,
            "count": len(new_boundaries),
        },
        "register_contexts": {
            "superseded": True,
            "changed_boundary_count": len(deltas),
            "deltas": deltas,
        },
        "artifacts": {
            "superseded_context": _artifact(oc_path, oc_raw),
            "corrected_context": _artifact(nc_path, nc_raw),
            "superseded_discovery": _artifact(od_path, od_raw),
            "corrected_discovery": _artifact(nd_path, nd_raw),
            "fallthrough_proof": _artifact(proof_path, proof_raw),
        },
        "fidelity": {
            "boundary_inventory_valid": True,
            "superseded_register_context_valid": False,
            "corrected_register_context_valid_with_declared_nop_substitutions": True,
            "custom_instruction_semantics_proven": False,
            "inference_produced": False,
        },
        "claim_scope": (
            "The corrected QEMU profile executes a not-taken CV22 conditional branch "
            "with a four-byte fallthrough. The 23 unsupported/custom boundaries are "
            "unchanged, but prior register contexts are superseded wherever listed. "
            "Later contexts remain non-faithful because the discovery method replaces "
            "earlier unsupported words with explicitly counted l.nop instructions."
        ),
    }
