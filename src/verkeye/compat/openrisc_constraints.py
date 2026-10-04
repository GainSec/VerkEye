"""Join static reachability, captured CPU state, and bounded CV22 models.

This ledger is intentionally conservative.  A captured register snapshot is
an input constraint, not proof of an instruction's state transition.  The
only bounded semantic model currently available covers two DVP helper
selectors; it is reported separately and never promoted to full hardware
fidelity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


class OpenRiscConstraintError(ValueError):
    """The joined compatibility evidence is malformed or contradictory."""


_SURFACE_SCHEMA = "verkeye.compat.openrisc-surface.v1"
_CONTEXT_SCHEMA = "verkeye.compat.openrisc-context.v1"
_DVP_SCHEMA = "verkeye.compat.cv22-dvp-evidence.v1"
_EXPECTED_DVP_OPERATIONS = (
    "opcode_0x07_function_0x002",
    "opcode_0x07_function_0x006",
)
_TAINTED_CONTEXT = re.compile(r"altered_by_(?P<count>\d+)_prior_nop_substitution")


def _canonical_sha256(document: Mapping[str, Any]) -> str:
    raw = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OpenRiscConstraintError(f"{field} must be an object")
    return value


def _require_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise OpenRiscConstraintError(f"{field} must be a list")
    return value


def _context_class(context: Mapping[str, Any]) -> tuple[str, int]:
    fidelity = context.get("pre_boundary_state_fidelity")
    substitutions = context.get("prior_nop_substitution_count")
    if fidelity == "exact_until_first_unsupported_instruction":
        if substitutions != 0:
            raise OpenRiscConstraintError(
                "exact context must have zero prior nop substitutions"
            )
        return "exact", 0
    if isinstance(fidelity, str):
        match = _TAINTED_CONTEXT.fullmatch(fidelity)
        if match is not None:
            expected = int(match.group("count"))
            if substitutions != expected or expected < 1:
                raise OpenRiscConstraintError(
                    "tainted context substitution count is contradictory"
                )
            return "tainted", expected
    raise OpenRiscConstraintError("context fidelity class is unsupported")


def _validate_registers(value: object) -> dict[str, str]:
    registers = _require_mapping(value, "register_state_before_boundary")
    expected = {f"r{index:02d}" for index in range(32)}
    if set(registers) != expected:
        raise OpenRiscConstraintError("context must contain registers r00-r31")
    result: dict[str, str] = {}
    for name in sorted(expected):
        word = registers[name]
        if not isinstance(word, str) or re.fullmatch(r"0x[0-9a-f]{8}", word) is None:
            raise OpenRiscConstraintError(f"{name} must be a canonical 32-bit hex word")
        result[name] = word
    return result


def _bounded_operation(occurrence: Mapping[str, Any]) -> str | None:
    if occurrence.get("opcode") != "0x07":
        return None
    fields = _require_mapping(
        occurrence.get("bit_fields_uninterpreted"),
        "bit_fields_uninterpreted",
    )
    selector = fields.get("bits_10_0")
    if selector == "0x002":
        return _EXPECTED_DVP_OPERATIONS[0]
    if selector == "0x006":
        return _EXPECTED_DVP_OPERATIONS[1]
    return None


def build_openrisc_constraint_ledger(
    surface: Mapping[str, Any],
    contexts: Mapping[str, Any],
    dvp_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a fail-closed ledger for each proven custom instruction site."""

    for document, schema, label in (
        (surface, _SURFACE_SCHEMA, "surface"),
        (contexts, _CONTEXT_SCHEMA, "contexts"),
        (dvp_evidence, _DVP_SCHEMA, "dvp_evidence"),
    ):
        if not isinstance(document, Mapping) or document.get("schema") != schema:
            raise OpenRiscConstraintError(f"{label} schema must be {schema}")

    surface_summary = _require_mapping(surface.get("summary"), "surface.summary")
    dynamic = _require_list(surface.get("dynamic_occurrences"), "dynamic_occurrences")
    static_only = _require_list(
        surface.get("static_only_proven_occurrences"),
        "static_only_proven_occurrences",
    )
    raw_occurrences = dynamic + static_only
    expected_count = surface_summary.get("proven_occurrence_count")
    if expected_count != len(raw_occurrences):
        raise OpenRiscConstraintError("surface proven occurrence count is contradictory")

    occurrences: dict[str, Mapping[str, Any]] = {}
    for raw in raw_occurrences:
        occurrence = _require_mapping(raw, "surface occurrence")
        address = occurrence.get("runtime_address")
        if not isinstance(address, str) or address in occurrences:
            raise OpenRiscConstraintError("surface occurrence address is invalid or duplicated")
        occurrences[address] = occurrence

    context_records: dict[str, Mapping[str, Any]] = {}
    for raw in _require_list(contexts.get("contexts"), "contexts.contexts"):
        context = _require_mapping(raw, "context")
        address = context.get("boundary_pc")
        if not isinstance(address, str) or address in context_records:
            raise OpenRiscConstraintError("context boundary is invalid or duplicated")
        occurrence = occurrences.get(address)
        instruction = _require_mapping(context.get("instruction"), "context.instruction")
        if occurrence is None or any(
            instruction.get(field) != occurrence.get(field)
            for field in ("runtime_address", "file_offset", "opcode", "word")
        ):
            raise OpenRiscConstraintError(
                f"context instruction does not match surface occurrence at {address}"
            )
        context_records[address] = context

    dynamic_addresses = {
        _require_mapping(item, "dynamic occurrence").get("runtime_address")
        for item in dynamic
    }
    if dynamic_addresses != set(context_records):
        raise OpenRiscConstraintError(
            "dynamic surface occurrences and captured contexts do not match"
        )

    implementation = _require_mapping(
        dvp_evidence.get("implementation_scope"),
        "dvp_evidence.implementation_scope",
    )
    implemented = implementation.get("implemented")
    if implemented != list(_EXPECTED_DVP_OPERATIONS):
        raise OpenRiscConstraintError("bounded DVP operation list is unexpected")
    bounded_model = {
        "implemented_operations": list(_EXPECTED_DVP_OPERATIONS),
        "hardware_defaults_emulated": implementation.get("hardware_defaults_emulated"),
        "unknown_reads": implementation.get("unknown_reads"),
        "all_other_custom_operations": implementation.get(
            "all_other_custom_operations"
        ),
    }

    ledger: list[dict[str, Any]] = []
    exact_count = 0
    tainted_count = 0
    bounded_count = 0
    for address in sorted(occurrences):
        occurrence = occurrences[address]
        operation = _bounded_operation(occurrence)
        if operation is not None:
            bounded_count += 1
        context = context_records.get(address)
        if context is None:
            dynamic_context: dict[str, Any] = {
                "available": False,
                "class": "not_observed",
            }
        else:
            context_class, substitutions = _context_class(context)
            if context_class == "exact":
                exact_count += 1
            else:
                tainted_count += 1
            dynamic_context = {
                "available": True,
                "class": context_class,
                "pre_boundary_state_fidelity": context[
                    "pre_boundary_state_fidelity"
                ],
                "prior_nop_substitution_count": substitutions,
                "register_state_before_boundary": _validate_registers(
                    context.get("register_state_before_boundary")
                ),
            }
        ledger.append(
            {
                "runtime_address": address,
                "file_offset": occurrence.get("file_offset"),
                "word": occurrence.get("word"),
                "opcode": occurrence.get("opcode"),
                "bit_fields_uninterpreted": occurrence.get(
                    "bit_fields_uninterpreted"
                ),
                "reachability": occurrence.get("reachability"),
                "dynamic_context": dynamic_context,
                "semantic_coverage": {
                    "status": "unresolved",
                    "bounded_model_operation": operation,
                },
            }
        )

    unresolved_count = sum(
        item["semantic_coverage"]["status"] == "unresolved" for item in ledger
    )
    summary = {
        "proven_occurrence_count": len(ledger),
        "dynamic_context_count": len(context_records),
        "exact_dynamic_context_count": exact_count,
        "tainted_dynamic_context_count": tainted_count,
        "static_only_occurrence_count": len(ledger) - len(context_records),
        "bounded_model_coverage_count": bounded_count,
        "unresolved_semantic_occurrence_count": unresolved_count,
        "inference_execution_supported": unresolved_count == 0,
    }
    return {
        "schema": "verkeye.compat.openrisc-constraint-ledger.v1",
        "sources": {
            "surface": {
                "schema": surface["schema"],
                "canonical_sha256": _canonical_sha256(surface),
            },
            "contexts": {
                "schema": contexts["schema"],
                "canonical_sha256": _canonical_sha256(contexts),
            },
            "dvp_evidence": {
                "schema": dvp_evidence["schema"],
                "canonical_sha256": _canonical_sha256(dvp_evidence),
            },
        },
        "bounded_model": bounded_model,
        "summary": summary,
        "occurrences": ledger,
        "gate": {
            "status": "blocked" if unresolved_count else "passed",
            "reason_codes": (
                ["CV22_CUSTOM_INSTRUCTION_SEMANTICS_UNRESOLVED"]
                if unresolved_count
                else []
            ),
            "unknown_computation_affecting_occurrence_count": unresolved_count,
        },
        "claim_scope": (
            "This ledger joins byte-verified reachability with QEMU exception-entry "
            "register snapshots and the bounded DVP probe model. A snapshot constrains "
            "instruction inputs but does not prove its state transition. Only the first "
            "dynamic snapshot is exact; later snapshots retain an explicit prior-nop "
            "taint count. No proven runtime occurrence is covered by the two modeled "
            "DVP selectors, so exact inference remains fail-closed."
        ),
    }
