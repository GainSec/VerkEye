#!/usr/bin/env python3
"""Verify one recovered NNCtrl load and CAVALRY_RUN_DAGS capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from verkeye.compat.cavalry import (  # noqa: E402
    CavalryError,
    analyze_run_dags_semantics,
    build_run_dags_command,
    parse_run_dags_snapshot,
    verify_run_dags_memory_topology,
    verify_run_dags_package_placement,
)
from verkeye.compat.driver_abi import analyze_cavalry_driver  # noqa: E402
from verkeye.compat.probe import (  # noqa: E402
    ProbeError,
    parse_probe_records,
    verify_probe_memory_dump,
)
from verkeye.cv22.container import parse_container  # noqa: E402
from verkeye.cv22.program import analyze_program_packages  # noqa: E402
from verkeye.cv22.tensors import parse_tensor_map  # noqa: E402
from verkeye.evidence import artifact_record, atomic_write_json  # noqa: E402


def _exact_record(
    records: tuple[dict[str, Any], ...],
    operation: str,
) -> dict[str, Any]:
    selected = [item for item in records if item.get("operation") == operation]
    if len(selected) != 1:
        raise ProbeError(
            f"expected exactly one {operation} probe record, observed {len(selected)}"
        )
    return selected[0]


def _parse_ioctl_trace(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CavalryError(
                f"invalid ioctl trace JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise CavalryError(f"ioctl trace line {line_number} is not an object")
        if record.get("schema") != "verkeye.cv22.ioctl-shim.v1":
            raise CavalryError(f"unsupported ioctl trace schema at line {line_number}")
        if record.get("sequence") != len(records):
            raise CavalryError("non-contiguous ioctl trace sequence")
        records.append(record)
    if not records:
        raise CavalryError("ioctl trace contains no records")
    return tuple(records)


def _exact_ioctl(
    records: tuple[dict[str, Any], ...],
    operation: str,
) -> dict[str, Any]:
    selected = [item for item in records if item.get("operation") == operation]
    if len(selected) != 1:
        raise CavalryError(
            f"expected exactly one {operation} ioctl record, observed {len(selected)}"
        )
    return selected[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--arena", required=True, type=Path)
    parser.add_argument("--probe-stdout", required=True, type=Path)
    parser.add_argument("--ioctl-trace", required=True, type=Path)
    parser.add_argument(
        "--semantic-source",
        type=Path,
        default=PROJECT / "third_party/ambarella/nnctrl-0.3.0/src/rundags.c",
    )
    parser.add_argument("--physical-base", required=True, type=lambda value: int(value, 0))
    parser.add_argument("--alignment", type=int, default=64)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    model = args.model.read_bytes()
    arena = args.arena.read_bytes()
    snapshot = parse_run_dags_snapshot(args.snapshot.read_bytes())
    driver = analyze_cavalry_driver(args.driver)
    probe_records = parse_probe_records(
        args.probe_stdout.read_text(encoding="utf-8")
    )
    ioctl_records = _parse_ioctl_trace(args.ioctl_trace)
    config = _exact_record(probe_records, "nnctrl_config")
    memory = _exact_record(probe_records, "nnctrl_memory")
    run = _exact_record(probe_records, "nnctrl_run_net")
    audio_clock = _exact_ioctl(ioctl_records, "CAVALRY_GET_AUDIO_CLK")
    run_dags = _exact_ioctl(ioctl_records, "CAVALRY_RUN_DAGS")
    driver_request = next(
        (
            item
            for item in driver["requests"]
            if item["request"] == 0xC0084303
        ),
        None,
    )
    if driver_request is None or driver_request["handler"] != "cavalry_run_dags":
        raise CavalryError("pinned driver lacks the recovered RUN_DAGS handler")

    if memory.get("physical_address") != args.physical_base:
        raise ProbeError(
            "probe/base physical address mismatch: "
            f"probe={memory.get('physical_address')}, argument={args.physical_base}"
        )
    expected_config_fields = (
        "total_dag_bytes",
        "input_bytes",
        "output_bytes",
        "working_bytes",
    )
    if any(
        not isinstance(config.get(field), int) for field in expected_config_fields
    ):
        raise ProbeError("probe config is missing an integer memory-size field")
    if run.get("status") != "failed" or run.get("return_code") != -1:
        raise ProbeError("capture must preserve the expected failed NNCtrl run")
    if (
        run_dags.get("outcome") != "captured_unsupported"
        or run_dags.get("errno") != 95
    ):
        raise CavalryError("RUN_DAGS trace did not fail closed with ENOTSUP")
    if (
        audio_clock.get("outcome") != "emulated"
        or audio_clock.get("audio_clock_hz") != 12_288_000
    ):
        raise CavalryError("audio-clock trace contradicts recovered value")

    splits = parse_tensor_map(model, parse_container(model))
    packages = analyze_program_packages(model, splits).packages
    placement = verify_run_dags_package_placement(
        snapshot,
        arena,
        physical_base=args.physical_base,
        model=model,
        packages=packages,
    )
    topology = verify_run_dags_memory_topology(
        snapshot,
        arena,
        placement,
        reported_total_dag_bytes=config["total_dag_bytes"],
        reported_input_bytes=config["input_bytes"],
        reported_output_bytes=config["output_bytes"],
        reported_working_bytes=config["working_bytes"],
        alignment=args.alignment,
    )
    memory_report = verify_probe_memory_dump(probe_records, args.arena)
    command = build_run_dags_command(snapshot, bytes(snapshot.command_size))
    descriptor_semantics = analyze_run_dags_semantics(
        snapshot,
        args.semantic_source.read_bytes(),
    )
    driver_contract = {
        "schema": "verkeye.cv22.run-dags-driver-contract.v1",
        "status": "verified_against_pinned_driver",
        "driver": artifact_record(args.driver),
        "provenance": {
            "copy_from_user": {
                "symbol": "cavalry_copy_dags_from_user",
                "virtual_address": 0x3F20,
                "size": 1_104,
            },
            "handler": {
                "symbol": "cavalry_run_dags",
                "virtual_address": 0x4BB0,
                "size": 1_296,
            },
            "kick": {
                "symbol": "visorc_kick_vp",
                "virtual_address": 0x2ED0,
                "size": 32,
            },
            "interrupt": {
                "symbol": "cavalry_vp_isr",
                "virtual_address": 0x01C4,
                "size": 172,
            },
            "status_decoder": {
                "symbol": "get_string_of_msg_code",
                "virtual_address": 0xB8F0,
                "size": 372,
            },
        },
        "request": {
            "ioctl": "0xc0084303",
            "handler": driver_request["handler"],
            "maximum_dags": 0x100,
            "descriptor_size": 0xB2C,
            "primary_table_maximum": 0x80,
            "primary_table_entry_size": 0x10,
            "secondary_table_maximum": 0x40,
            "secondary_table_entry_size": 0x0C,
            "timeout_milliseconds": snapshot.timeout_milliseconds,
        },
        "validation": {
            "required_nonzero_descriptor_word_offsets": [
                "0x04",
                "0x08",
                "0x0c",
                "0x10",
            ],
            "required_nonzero_primary_entry_word_offset": "0x08",
            "long_timeout_control_halfword_offset": "0x02",
            "long_timeout_when_greater_than": 1,
        },
        "command": {
            "initial_state": "all_zero_witness",
            "size": len(command),
            "sha256": hashlib.sha256(command).hexdigest(),
            "header_size": 0x30,
            "command_code": 1,
            "dag_count": len(snapshot.descriptors),
            "descriptor_destination_offset": 0x30,
            "fixed_descriptor_copy_bytes": 0x2C,
            "primary_table_destination_offset": 0x2C,
            "secondary_table_destination_offset": 0x82C,
            "unwritten_bytes_preserved": True,
        },
        "mailbox": {
            "kick_function_virtual_address": 0x2ED0,
            "store_barrier": "dsb st",
            "mmio_offset": 0x5F1FC,
            "written_value": 0x1400,
            "run_completion_flag_device_offset": 0x3E0,
            "run_response_magic": "0x80000001",
            "stop_response_magic": "0x80000002",
        },
        "response": {
            "minimum_bytes": 0x14,
            "magic_offset": 0x00,
            "status_offset": 0x04,
            "additional_result_word_offsets": ["0x08", "0x0c", "0x10"],
            "copied_header_bytes": 0x10,
            "descriptor_result_source_offset": 0x14,
            "descriptor_result_destination_offset": 0x14,
            "firmware_error_ioctl_result": -53,
        },
    }

    report = {
        "schema": "verkeye.cv22.cavalry-capture-analysis.v1",
        "status": "verified_pre_execution_layout",
        "inputs": {
            "model": artifact_record(args.model),
            "driver": artifact_record(args.driver),
            "snapshot": artifact_record(args.snapshot),
            "arena": artifact_record(args.arena),
            "probe_stdout": artifact_record(args.probe_stdout),
            "ioctl_trace": artifact_record(args.ioctl_trace),
            "semantic_source": artifact_record(args.semantic_source),
        },
        "probe_config": {
            field: config[field] for field in expected_config_fields
        },
        "probe_memory": memory_report,
        "package_placement": placement.to_document(),
        "memory_topology": topology.to_document(),
        "driver_contract": driver_contract,
        "descriptor_semantics": descriptor_semantics,
        "execution": {
            "status": "captured_unsupported",
            "request": run_dags["request"],
            "errno": run_dags["errno"],
            "audio_clock_hz": audio_clock["audio_clock_hz"],
            "inference_produced": False,
            "fidelity_boundary": (
                "exact recovered userspace parsing and pre-execution layout are "
                "verified; accelerator computation remains unsupported"
            ),
        },
    }
    atomic_write_json(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
