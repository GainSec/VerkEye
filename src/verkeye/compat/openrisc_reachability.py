"""Evidence-scoped OpenRISC control-flow and opcode-family inventory.

The recovered CV22 firmware is a raw binary that contains executable code,
tables, strings, and padding in one disassembly.  Treating every row emitted by
``objdump`` as code therefore creates false opcode matches.  This module walks
only standard, explicit OpenRISC control-flow edges from caller-supplied entry
points and byte-verified dynamic observations.

Implementation-defined instructions are recorded but never interpreted.  The
walker continues their linear successor as *conditional* evidence so useful
context is retained without claiming that the unknown instruction falls
through.  Indirect control transfers stop the affected path and are reported.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .openrisc_static import Instruction, OpenRiscStaticError, parse_objdump


class OpenRiscReachabilityError(ValueError):
    """Raised when reachability inputs are malformed or contradict evidence."""


_HEX32 = re.compile(r"^0x[0-9a-fA-F]{1,8}$")
_DIRECT_JUMP = re.compile(r"^l\.(?P<kind>j|jal)\s+0x(?P<target>[0-9a-fA-F]+)$")
_CONDITIONAL_BRANCH = re.compile(
    r"^l\.(?P<kind>bf|bnf)\s+0x(?P<target>[0-9a-fA-F]+)$"
)
_INDIRECT_JUMP = re.compile(r"^l\.(?P<kind>jr|jalr)\s+(?P<register>r\d+)$")

_PROVEN = "proven"
_CONDITIONAL = "conditional_after_unresolved_instruction"


def _hex32(value: int) -> str:
    return f"0x{value:08x}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(name: str, data: bytes) -> dict[str, object]:
    return {"name": name, "size": len(data), "sha256": _sha256(data)}


def _parse_hex32(value: object, *, field: str) -> int:
    if not isinstance(value, str) or _HEX32.fullmatch(value) is None:
        raise OpenRiscReachabilityError(f"{field} must be a hexadecimal 32-bit value")
    return int(value, 16)


def _validate_uint32(value: object, *, field: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 0xFFFF_FFFF
    ):
        raise OpenRiscReachabilityError(f"{field} must be a 32-bit integer")
    return value


def _weaken(certainty: str) -> str:
    return _CONDITIONAL if certainty == _PROVEN else certainty


def _public_instruction(
    item: Instruction,
    *,
    runtime_base: int,
    certainty: str | None,
    reasons: Iterable[str] = (),
) -> dict[str, object]:
    return {
        "runtime_address": _hex32(item.address),
        "file_offset": _hex32(item.address - runtime_base),
        "bytes_little_endian": item.raw.hex(),
        "word": _hex32(item.word),
        "opcode": f"0x{item.word >> 26:02x}",
        "disassembly": item.text,
        "reachability": certainty or "excluded_not_reached",
        "reasons": sorted(set(reasons)),
    }


def _verify_disassembly(
    firmware: bytes,
    instructions: Sequence[Instruction],
    *,
    runtime_base: int,
) -> None:
    for item in instructions:
        offset = item.address - runtime_base
        if offset < 0 or offset + 4 > len(firmware):
            raise OpenRiscReachabilityError(
                f"disassembly address {_hex32(item.address)} is outside firmware"
            )
        if firmware[offset : offset + 4] != item.raw:
            raise OpenRiscReachabilityError(
                f"disassembly at {_hex32(item.address)} differs from firmware"
            )


def _dynamic_observations(
    boundaries: Mapping[str, Any], by_address: Mapping[int, Instruction]
) -> tuple[int, ...]:
    raw = boundaries.get("dynamically_reached_instructions")
    if not isinstance(raw, list):
        raise OpenRiscReachabilityError(
            "boundary report lacks dynamically_reached_instructions"
        )
    observations: list[int] = []
    seen: set[int] = set()
    for index, record in enumerate(raw):
        if not isinstance(record, Mapping):
            raise OpenRiscReachabilityError(f"boundary {index} is not an object")
        address = _parse_hex32(record.get("runtime_address"), field=f"boundary {index} address")
        expected = _parse_hex32(record.get("word"), field=f"boundary {index} word")
        if address in seen:
            raise OpenRiscReachabilityError(
                f"duplicate dynamic observation at {_hex32(address)}"
            )
        seen.add(address)
        instruction = by_address.get(address)
        if instruction is None:
            raise OpenRiscReachabilityError(
                f"dynamic observation {_hex32(address)} is absent from disassembly"
            )
        if instruction.word != expected:
            raise OpenRiscReachabilityError(
                f"dynamic observation {_hex32(address)} word differs from disassembly"
            )
        observations.append(address)
    return tuple(observations)


def analyze_openrisc_reachability(
    firmware_path: str | Path,
    disassembly_path: str | Path,
    boundaries: Mapping[str, Any],
    *,
    runtime_base: int = 0x0040_0000,
    entry_points: Sequence[int] = (0x0040_0000,),
    code_ranges: Sequence[tuple[int, int]] = (),
    family_opcode: int = 0x10,
    context_instructions: int = 4,
    include_instruction_inventory: bool = True,
    delay_slots: bool = True,
) -> dict[str, Any]:
    """Inventory an opcode family without promoting raw data to proven code.

    ``code_ranges`` are optional, half-open, explicitly asserted ranges.  They
    are never inferred.  Their instructions are seeded with a distinct reason
    so the report preserves the origin of that trust decision.
    """

    runtime_base = _validate_uint32(runtime_base, field="runtime_base")
    family_opcode = _validate_uint32(family_opcode, field="family_opcode")
    if family_opcode > 0x3F:
        raise OpenRiscReachabilityError("family_opcode must fit six bits")
    if (
        not isinstance(context_instructions, int)
        or isinstance(context_instructions, bool)
        or context_instructions < 0
    ):
        raise OpenRiscReachabilityError("context_instructions must be non-negative")
    if not isinstance(boundaries, Mapping):
        raise OpenRiscReachabilityError("boundaries must be an object")
    if not isinstance(include_instruction_inventory, bool):
        raise OpenRiscReachabilityError(
            "include_instruction_inventory must be boolean"
        )
    if not isinstance(delay_slots, bool):
        raise OpenRiscReachabilityError("delay_slots must be boolean")

    branch_fallthrough_bytes = 8 if delay_slots else 4

    firmware_source = Path(firmware_path).resolve()
    disassembly_source = Path(disassembly_path).resolve()
    try:
        firmware = firmware_source.read_bytes()
        raw_disassembly = disassembly_source.read_bytes()
        instructions = parse_objdump(raw_disassembly)
    except (OSError, OpenRiscStaticError) as exc:
        raise OpenRiscReachabilityError(f"cannot load reachability input: {exc}") from exc

    _verify_disassembly(firmware, instructions, runtime_base=runtime_base)
    by_address = {item.address: item for item in instructions}
    address_to_index = {item.address: index for index, item in enumerate(instructions)}
    dynamic = _dynamic_observations(boundaries, by_address)

    seeds: list[tuple[int, str]] = []
    for index, value in enumerate(entry_points):
        address = _validate_uint32(value, field=f"entry_points[{index}]")
        if address not in by_address:
            raise OpenRiscReachabilityError(
                f"entry point {_hex32(address)} is absent from disassembly"
            )
        seeds.append((address, "explicit_entry_point"))
    for address in dynamic:
        seeds.append((address, "dynamic_observation"))

    for index, value in enumerate(code_ranges):
        if not isinstance(value, tuple) or len(value) != 2:
            raise OpenRiscReachabilityError(f"code_ranges[{index}] must be a pair")
        start = _validate_uint32(value[0], field=f"code_ranges[{index}].start")
        end = _validate_uint32(value[1], field=f"code_ranges[{index}].end")
        if start % 4 or end % 4 or start >= end:
            raise OpenRiscReachabilityError(
                f"code_ranges[{index}] must be aligned and non-empty"
            )
        matching = [item.address for item in instructions if start <= item.address < end]
        if not matching or matching[0] != start or matching[-1] != end - 4:
            raise OpenRiscReachabilityError(
                f"code_ranges[{index}] is not fully represented in disassembly"
            )
        seeds.extend((address, "explicit_code_range") for address in matching)

    if not seeds:
        raise OpenRiscReachabilityError("at least one evidence seed is required")

    certainty: dict[int, str] = {}
    reasons: defaultdict[int, set[str]] = defaultdict(set)
    work: deque[tuple[int, str]] = deque()
    flow_enabled: dict[int, bool] = {}
    unresolved: dict[tuple[int, str], dict[str, object]] = {}

    def record_unresolved(
        item: Instruction,
        kind: str,
        current_certainty: str,
        detail: str,
    ) -> None:
        unresolved[(item.address, kind)] = {
            "runtime_address": _hex32(item.address),
            "word": _hex32(item.word),
            "disassembly": item.text,
            "kind": kind,
            "reachability": current_certainty,
            "detail": detail,
        }

    def enqueue(
        address: int,
        next_certainty: str,
        reason: str,
        *,
        continue_flow: bool = True,
    ) -> None:
        if address not in by_address:
            # The source instruction is not always available here.  A synthetic
            # unresolved edge is represented under the missing target itself.
            unresolved[(address, "missing_target")] = {
                "runtime_address": _hex32(address),
                "word": None,
                "disassembly": None,
                "kind": "missing_target",
                "reachability": next_certainty,
                "detail": "control-flow target is absent from byte-verified disassembly",
            }
            return
        reasons[address].add(reason)
        current = certainty.get(address)
        current_flow = flow_enabled.get(address, False)
        improves_certainty = current != _PROVEN and current != next_certainty
        enables_flow = continue_flow and not current_flow
        if not improves_certainty and not enables_flow:
            return
        if current != _PROVEN:
            certainty[address] = next_certainty
        flow_enabled[address] = current_flow or continue_flow
        if flow_enabled[address]:
            work.append((address, certainty[address]))

    for address, reason in seeds:
        enqueue(address, _PROVEN, reason)

    while work:
        address, queued_certainty = work.popleft()
        if certainty.get(address) != queued_certainty:
            continue
        item = by_address[address]
        text = item.text

        direct = _DIRECT_JUMP.fullmatch(text)
        conditional = _CONDITIONAL_BRANCH.fullmatch(text)
        indirect = _INDIRECT_JUMP.fullmatch(text)
        if direct is not None or conditional is not None or indirect is not None:
            successor_certainty = queued_certainty
            if delay_slots:
                delay_address = address + 4
                delay_item = by_address.get(delay_address)
                if delay_item is None:
                    record_unresolved(
                        item,
                        "missing_delay_slot",
                        queued_certainty,
                        "control transfer has no byte-verified delay-slot instruction",
                    )
                    successor_certainty = _weaken(successor_certainty)
                else:
                    enqueue(
                        delay_address,
                        queued_certainty,
                        "branch_delay_slot",
                        continue_flow=False,
                    )
                    if delay_item.text == "*unknown*":
                        successor_certainty = _weaken(successor_certainty)

            if direct is not None:
                target = int(direct.group("target"), 16)
                kind = direct.group("kind")
                enqueue(
                    target,
                    successor_certainty,
                    "direct_call_target" if kind == "jal" else "direct_jump_target",
                )
                if kind == "jal":
                    enqueue(
                        address + branch_fallthrough_bytes,
                        successor_certainty,
                        "direct_call_fallthrough",
                    )
                continue
            if conditional is not None:
                target = int(conditional.group("target"), 16)
                enqueue(target, successor_certainty, "conditional_branch_target")
                enqueue(
                    address + branch_fallthrough_bytes,
                    successor_certainty,
                    "conditional_branch_fallthrough",
                )
                continue

            assert indirect is not None
            register = indirect.group("register")
            kind = indirect.group("kind")
            if kind == "jr" and register == "r9":
                continue
            record_unresolved(
                item,
                "indirect_call" if kind == "jalr" else "indirect_jump",
                queued_certainty,
                f"target register {register} is unresolved",
            )
            continue

        if text == "*unknown*":
            record_unresolved(
                item,
                "unresolved_instruction_semantics",
                queued_certainty,
                "implementation-defined instruction may alter control flow",
            )
            enqueue(
                address + 4,
                _weaken(queued_certainty),
                "linear_successor_after_unresolved_instruction",
            )
            continue

        if text.startswith(("l.rfe", "l.sys", "l.trap")):
            record_unresolved(
                item,
                "system_control_transfer",
                queued_certainty,
                "exception or system-control successor is unresolved",
            )
            continue

        enqueue(address + 4, queued_certainty, "linear_fallthrough")

    reached = [item for item in instructions if item.address in certainty]
    family = [item for item in reached if item.word >> 26 == family_opcode]
    family_occurrences: list[dict[str, object]] = []
    for item in family:
        index = address_to_index[item.address]
        start = max(0, index - context_instructions)
        end = min(len(instructions), index + context_instructions + 1)
        public = _public_instruction(
            item,
            runtime_base=runtime_base,
            certainty=certainty[item.address],
            reasons=reasons[item.address],
        )
        occurrence_reasons = public.pop("reasons")
        public["reachability"] = {
            "certainty": certainty[item.address],
            "reasons": occurrence_reasons,
        }
        family_occurrences.append(
            {
                **public,
                "bit_fields_uninterpreted": {
                    "bits_25_21": (item.word >> 21) & 0x1F,
                    "bits_20_16": (item.word >> 16) & 0x1F,
                    "bits_15_11": (item.word >> 11) & 0x1F,
                    "bits_10_0": f"0x{item.word & 0x7FF:03x}",
                },
                "semantics": "unresolved",
                "context": [
                    _public_instruction(
                        context,
                        runtime_base=runtime_base,
                        certainty=certainty.get(context.address),
                        reasons=reasons.get(context.address, ()),
                    )
                    for context in instructions[start:end]
                ],
            }
        )

    excluded = [
        {
            "runtime_address": _hex32(item.address),
            "word": _hex32(item.word),
            "reason": "not_reached_from_any_evidence_seed",
        }
        for item in instructions
        if item.word >> 26 == family_opcode and item.address not in certainty
    ]
    family_words = sorted(
        {item.word for item in instructions if item.word >> 26 == family_opcode}
    )
    family_variant_summary = []
    for word in family_words:
        matching = [item for item in instructions if item.word == word]
        proven = sum(
            certainty.get(item.address) == _PROVEN for item in matching
        )
        conditional = sum(
            certainty.get(item.address) == _CONDITIONAL for item in matching
        )
        family_variant_summary.append(
            {
                "word": _hex32(word),
                "proven_count": proven,
                "conditional_count": conditional,
                "excluded_count": len(matching) - proven - conditional,
                "total_disassembly_count": len(matching),
            }
        )
    unresolved_output = [unresolved[key] for key in sorted(unresolved)]
    proven_count = sum(certainty[item.address] == _PROVEN for item in reached)
    conditional_count = len(reached) - proven_count
    proven_family_count = sum(certainty[item.address] == _PROVEN for item in family)

    report: dict[str, Any] = {
        "schema": "verkeye.compat.openrisc-reachability.v1",
        "artifact": _artifact(firmware_source.name, firmware),
        "disassembly_artifact": _artifact(disassembly_source.name, raw_disassembly),
        "runtime_base": _hex32(runtime_base),
        "control_flow_profile": {
            "delay_slots": delay_slots,
            "direct_call_return_bytes": branch_fallthrough_bytes,
            "conditional_fallthrough_bytes": branch_fallthrough_bytes,
        },
        "evidence_seeds": {
            "entry_points": [_hex32(address) for address in entry_points],
            "dynamic_observations": [_hex32(address) for address in dynamic],
            "explicit_code_ranges": [
                {"start": _hex32(start), "end_exclusive": _hex32(end)}
                for start, end in code_ranges
            ],
        },
        "summary": {
            "instruction_count": len(reached),
            "proven_instruction_count": proven_count,
            "conditional_instruction_count": conditional_count,
            "family_opcode": f"0x{family_opcode:02x}",
            "proven_family_occurrence_count": proven_family_count,
            "conditional_family_occurrence_count": len(family) - proven_family_count,
            "unresolved_control_transfer_count": len(unresolved_output),
            "excluded_disassembly_row_count": len(excluded),
            "state_transitions_resolved": False,
            "inference_execution_supported": False,
        },
        "instruction_inventory_included": include_instruction_inventory,
        "family_occurrences": family_occurrences,
        "family_variant_summary": family_variant_summary,
        "excluded_disassembly_rows": excluded,
        "unresolved_control_transfers": unresolved_output,
        "claim_scope": (
            "This report identifies byte-verified instructions reached through "
            "standard OpenRISC control-flow edges, explicit entry/range assertions, "
            "or dynamic observations. It does not assign semantics to implementation-"
            "defined words. Linear successors after such words remain conditional, "
            "and indirect successors and unreachable opcode-like data remain excluded. "
            f"Control transfers use {'architectural delay slots' if delay_slots else 'the CV22 no-delay profile'}."
        ),
    }
    if include_instruction_inventory:
        report["instructions"] = [
            _public_instruction(
                item,
                runtime_base=runtime_base,
                certainty=certainty[item.address],
                reasons=reasons[item.address],
            )
            for item in reached
        ]
    return report
