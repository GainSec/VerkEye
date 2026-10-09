from __future__ import annotations

from pathlib import Path

import pytest

from verkeye.compat.ades_runtime import (
    AdesRuntimeError,
    build_capture_argv,
    build_prepare_argv,
    build_run_argv,
    load_runtime_spec,
)


SPEC = Path("config/cb62-ades-runtime.json")
MODEL = Path("fixtures/models/yolov6n_hor.bin")


def test_runtime_spec_pins_exact_model_toolchain_and_all_dvi_hashes() -> None:
    spec = load_runtime_spec(SPEC)

    assert spec.model_sha256 == (
        "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"
    )
    assert spec.image == (
        "xiezhouyi/cnngen@sha256:"
        "4c4266444740d9d52f6dd695fff86763ebb51dfb01b0c9778ca20209308cdcdb"
    )
    assert spec.platform == "linux/amd64"
    assert spec.cavalry_version == "2.1.7"
    assert spec.cavalry_hash == "c5db5f1"
    assert len(spec.dvi_sha256) == 10
    assert spec.verify_model(MODEL) == spec.model_sha256


def test_runtime_spec_rejects_modified_model(tmp_path: Path) -> None:
    spec = load_runtime_spec(SPEC)
    modified = tmp_path / "modified.bin"
    modified.write_bytes(MODEL.read_bytes() + b"modified")

    with pytest.raises(AdesRuntimeError, match="model SHA-256"):
        spec.verify_model(modified)


def test_prepare_command_is_digest_pinned_and_generates_exact_dvi_contract(
    tmp_path: Path,
) -> None:
    spec = load_runtime_spec(SPEC)
    argv = build_prepare_argv(spec, tmp_path)

    assert argv[:6] == (
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--volume",
    )
    assert argv[6] == f"{tmp_path.resolve()}:/work"
    assert spec.image in argv
    command = argv[-1]
    assert "source /usr/local/amba-cv-tools-2.4.2.0.1073.ubuntu-18.04/env/cv22.env" in command
    assert "cavalry_gen -V 2.1.7 -f /work/model.bin" in command
    assert "-p /work -v" in command
    assert "g++ -std=c++17" in command
    assert "/work/ades_executor.cpp" in command


def test_run_command_uses_only_prepared_assets_and_bounded_output_directory(
    tmp_path: Path,
) -> None:
    spec = load_runtime_spec(SPEC)
    argv = build_run_argv(spec, tmp_path, "run-000001")

    assert argv[:6] == (
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--volume",
    )
    assert argv[6] == f"{tmp_path.resolve()}:/work"
    command = argv[-1]
    assert "/work/ades-executor" in command
    assert "/work/manifest.tsv" in command
    assert "/work/parse" in command
    assert "/work/run-000001/input.tensor" in command
    assert "/work/run-000001" in command
    assert "stdout.tsv" in command
    assert "stderr.log" in command


def test_capture_command_compiles_hooks_and_runs_bounded_zero_input(
    tmp_path: Path,
) -> None:
    spec = load_runtime_spec(SPEC)

    argv = build_capture_argv(spec, tmp_path)

    assert argv[:6] == (
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--volume",
    )
    command = argv[-1]
    assert "ades_kernel_capture.cpp" in command
    assert "ades_operator_capture.cpp" in command
    assert "g++ -std=c++17 -shared -fPIC -O2" in command
    assert "VERKEYE_FASTCONV_DUMP=/work/capture/fastconv" in command
    assert "VERKEYE_OPERATOR_DUMP=/work/capture/operators" in command
    assert "LD_PRELOAD=/work/ades_kernel_capture.so:/work/ades_operator_capture.so" in command
    assert "head -c 1984512 /dev/zero > /work/capture/input.tensor" in command
    assert "/work/ades-executor" in command
    assert "rm -rf" not in command
    assert "capture workspace already exists" in command
