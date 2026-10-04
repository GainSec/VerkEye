"""Aggregate the evidenced CV22 implementation-defined instruction surface.

The scheduler image mixes executable code and data.  This analyzer therefore
builds on the byte-verified OpenRISC control-flow walk rather than scanning all
32-bit words and pretending opcode-shaped data is code.  It separates:

* proven occurrences reached before any unresolved instruction;
* conditional occurrences lying after an unresolved instruction; and
* opcode-shaped disassembly rows excluded from every evidence seed.

No custom operand layout or execution behavior is assigned here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .openrisc_reachability import (
    OpenRiscReachabilityError,
    analyze_openrisc_reachability,
)


class OpenRiscSurfaceError(ValueError):
    """The instruction-surface inputs are malformed or contradictory."""


_FAMILIES = (
    (0x07, "unallocated_orbis32_opcode_0x07"),
    (0x10, "unallocated_orbis32_opcode_0x10"),
    (0x1C, "l.cust1"),
    (0x1F, "l.cust4"),
)


def _same(field: str, reports: list[dict[str, Any]]) -> Any:
    first = reports[0][field]
    if any(report[field] != first for report in reports[1:]):
        raise OpenRiscSurfaceError(
            f"per-family reachability reports disagree on {field}"
        )
    return first


def analyze_openrisc_surface(
    firmware_path: str | Path,
    disassembly_path: str | Path,
    boundaries: Mapping[str, Any],
    *,
    runtime_base: int = 0x0040_0000,
) -> dict[str, Any]:
    """Return a fail-closed inventory for all four observed custom families."""

    reports: list[dict[str, Any]] = []
    try:
        for opcode, _name in _FAMILIES:
            reports.append(
                analyze_openrisc_reachability(
                    firmware_path,
                    disassembly_path,
                    boundaries,
                    runtime_base=runtime_base,
                    family_opcode=opcode,
                    include_instruction_inventory=False,
                    delay_slots=False,
                )
            )
    except OpenRiscReachabilityError as exc:
        raise OpenRiscSurfaceError(str(exc)) from exc

    artifact = _same("artifact", reports)
    disassembly_artifact = _same("disassembly_artifact", reports)
    control_flow_profile = _same("control_flow_profile", reports)
    evidence_seeds = _same("evidence_seeds", reports)
    unresolved_control_transfers = _same("unresolved_control_transfers", reports)

    families: list[dict[str, Any]] = []
    all_proven: list[dict[str, Any]] = []
    all_conditional: list[dict[str, Any]] = []
    excluded_count = 0
    unique_variant_count = 0
    for (opcode, name), report in zip(_FAMILIES, reports, strict=True):
        proven = [
            item
            for item in report["family_occurrences"]
            if item["reachability"]["certainty"] == "proven"
        ]
        conditional = [
            item
            for item in report["family_occurrences"]
            if item["reachability"]["certainty"]
            == "conditional_after_unresolved_instruction"
        ]
        unexpected = len(report["family_occurrences"]) - len(proven) - len(conditional)
        if unexpected:
            raise OpenRiscSurfaceError(
                f"opcode 0x{opcode:02x} contains an unknown certainty class"
            )
        summary = report["summary"]
        family_excluded = summary["excluded_disassembly_row_count"]
        variants = report["family_variant_summary"]
        families.append(
            {
                "opcode": f"0x{opcode:02x}",
                "family": name,
                "unique_variant_count": len(variants),
                "proven_occurrence_count": len(proven),
                "conditional_occurrence_count": len(conditional),
                "excluded_disassembly_row_count": family_excluded,
                "proven_occurrences": proven,
                "conditional_occurrences": conditional,
                "variant_summary": variants,
            }
        )
        all_proven.extend(proven)
        all_conditional.extend(conditional)
        excluded_count += family_excluded
        unique_variant_count += len(variants)

    all_proven.sort(key=lambda item: int(item["runtime_address"], 16))
    all_conditional.sort(key=lambda item: int(item["runtime_address"], 16))
    dynamic = [
        item
        for item in all_proven
        if "dynamic_observation" in item["reachability"]["reasons"]
    ]
    static_only = [
        item
        for item in all_proven
        if "dynamic_observation" not in item["reachability"]["reasons"]
    ]

    return {
        "schema": "verkeye.compat.openrisc-surface.v1",
        "artifact": artifact,
        "disassembly_artifact": disassembly_artifact,
        "runtime_base": f"0x{runtime_base:08x}",
        "control_flow_profile": control_flow_profile,
        "evidence_seeds": evidence_seeds,
        "summary": {
            "family_count": len(families),
            "unique_variant_count": unique_variant_count,
            "proven_occurrence_count": len(all_proven),
            "dynamic_occurrence_count": len(dynamic),
            "static_only_proven_occurrence_count": len(static_only),
            "conditional_occurrence_count": len(all_conditional),
            "excluded_disassembly_row_count": excluded_count,
            "state_transitions_resolved": False,
            "inference_execution_supported": False,
        },
        "families": families,
        "dynamic_occurrences": dynamic,
        "static_only_proven_occurrences": static_only,
        "conditionally_reached_occurrences": all_conditional,
        "unresolved_control_transfers": unresolved_control_transfers,
        "gate": {
            "status": "blocked",
            "reason_codes": ["CV22_CUSTOM_INSTRUCTION_SEMANTICS_UNRESOLVED"],
            "unknown_computation_affecting_occurrence_count": len(all_proven),
        },
        "claim_scope": (
            "This report aggregates four implementation-defined opcode families "
            "under the corrected CV22 no-delay control-flow profile. Proven means "
            "standard control flow or an exact dynamic observation reaches the word "
            "without first traversing an unresolved instruction. Conditional means "
            "the path already crossed an unresolved instruction and is not proof of "
            "execution. Excluded rows may be code or data and are not promoted. No "
            "operand layout, state transition, or instruction behavior is inferred."
        ),
    }

