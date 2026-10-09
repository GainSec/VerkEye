from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from types import MappingProxyType

import pytest

from verkeye.cv22.parameter_extraction import (
    FastconvCapture,
    FastconvChannel,
    SparseKernelPoint,
    encode_fastconv_capture,
)
from verkeye.runtime.generation import (
    GeneratedRuntimeError,
    assemble_generated_runtime,
    generate_owner_runtime,
    generated_runtime_path,
    load_generated_runtime,
)
from verkeye.compat.ades_runtime import AdesRuntimeSpec, load_runtime_spec


MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_runtime(root: Path, *, calls: int = 55) -> None:
    members: list[dict[str, object]] = []
    for call in range(calls):
        payload = b"VKFC" + call.to_bytes(4, "little")
        relative = f"fastconv/call-{call:03d}.vkfc"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        members.append(
            {
                "call_index": call,
                "path": relative,
                "size": len(payload),
                "sha256": _sha256(payload),
            }
        )
    masks: dict[str, dict[str, object]] = {}
    for name, size, height, width in (
        ("split4", 342, 38, 68),
        ("split5", 1292, 76, 136),
    ):
        payload = bytes(size)
        relative = f"masks/{name}.bin"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        masks[name] = {
            "path": relative,
            "size": size,
            "sha256": _sha256(payload),
            "height": height,
            "width": width,
        }
    document = {
        "schema": "verkeye.generated-runtime.v1",
        "model_sha256": MODEL_SHA256,
        "toolchain": {
            "image": "example.invalid/ades@sha256:" + "a" * 64,
            "platform": "linux/amd64",
        },
        "fastconv": members,
        "masks": masks,
    }
    (root / "manifest.json").write_text(json.dumps(document), encoding="utf-8")


def test_generated_runtime_path_is_content_addressed(tmp_path: Path) -> None:
    assert generated_runtime_path(tmp_path, MODEL_SHA256) == tmp_path / MODEL_SHA256


def test_load_generated_runtime_verifies_every_member(tmp_path: Path) -> None:
    _write_runtime(tmp_path)
    runtime = load_generated_runtime(tmp_path, expected_model_sha256=MODEL_SHA256)

    assert runtime.model_sha256 == MODEL_SHA256
    assert len(runtime.fastconv) == 55
    assert runtime.fastconv[54].call_index == 54
    assert runtime.split4_mask == tmp_path / "masks/split4.bin"
    assert runtime.split5_mask == tmp_path / "masks/split5.bin"


def test_load_generated_runtime_rejects_model_mismatch(tmp_path: Path) -> None:
    _write_runtime(tmp_path)
    with pytest.raises(GeneratedRuntimeError, match="model SHA-256 mismatch"):
        load_generated_runtime(tmp_path, expected_model_sha256="0" * 64)


def test_load_generated_runtime_rejects_missing_fastconv_call(tmp_path: Path) -> None:
    _write_runtime(tmp_path, calls=54)
    with pytest.raises(GeneratedRuntimeError, match="exactly 55"):
        load_generated_runtime(tmp_path)


def test_load_generated_runtime_rejects_corrupt_member(tmp_path: Path) -> None:
    _write_runtime(tmp_path)
    (tmp_path / "fastconv/call-017.vkfc").write_bytes(b"changed!")
    with pytest.raises(GeneratedRuntimeError, match="SHA-256 mismatch"):
        load_generated_runtime(tmp_path)


def test_load_generated_runtime_rejects_member_path_traversal(tmp_path: Path) -> None:
    _write_runtime(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["fastconv"][0]["path"] = "../outside.vkfc"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(GeneratedRuntimeError, match="relative member path"):
        load_generated_runtime(tmp_path)


def test_load_generated_runtime_rejects_symlinked_member_parent(
    tmp_path: Path,
) -> None:
    _write_runtime(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    payload = (tmp_path / "fastconv/call-000.vkfc").read_bytes()
    (outside / "call-000.vkfc").write_bytes(payload)
    (tmp_path / "links").symlink_to(outside, target_is_directory=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["fastconv"][0]["path"] = "links/call-000.vkfc"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(GeneratedRuntimeError, match="outside runtime root"):
        load_generated_runtime(tmp_path)


def test_load_generated_runtime_rejects_wrong_mask_contract(tmp_path: Path) -> None:
    _write_runtime(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["masks"]["split4"]["width"] = 69
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(GeneratedRuntimeError, match="split4 mask contract"):
        load_generated_runtime(tmp_path)


def test_generated_runtime_decodes_normalized_fastconv_member(tmp_path: Path) -> None:
    _write_runtime(tmp_path)
    capture = FastconvCapture(
        channels=(
            FastconvChannel(
                points=(SparseKernelPoint(0, 0, 0, -7),),
                accumulator_shift=8,
                offset=3,
                output_saturation_max=255,
                final_shift_control=2,
                points_sha256=_sha256((0).to_bytes(1, "little")),
            ),
        ),
        entries_sha256="0" * 64,
    )
    payload = encode_fastconv_capture(capture)
    package = tmp_path / "fastconv/call-000.vkfc"
    package.write_bytes(payload)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["fastconv"][0]["size"] = len(payload)
    manifest["fastconv"][0]["sha256"] = _sha256(payload)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    runtime = load_generated_runtime(tmp_path)

    assert runtime.fastconv_capture(0).channels[0].points[0].weight == -7


def _write_raw_capture(root: Path) -> None:
    fastconv = root / "fastconv"
    masks = root / "masks"
    fastconv.mkdir(parents=True)
    masks.mkdir()
    entry = struct.pack(
        "<12Q", 0x1000, 0x1000, 0x1000, 0, 0, 0, 0, 8, 0, 255, 2, 0
    )
    for call in range(55):
        (fastconv / f"fastconv-{call}-entries.bin").write_bytes(entry)
    split4 = bytearray([0x55] * 342)
    split5 = bytes([0xAA] * 1292)
    (masks / "import-135-slot-0.bin").write_bytes(split4)
    (masks / "import-175-slot-0.bin").write_bytes(split5)


def test_assemble_generated_runtime_normalizes_and_atomically_installs(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture"
    _write_raw_capture(capture)
    output = tmp_path / "generated"
    spec = load_runtime_spec(Path("config/cb62-ades-runtime.json"))

    runtime = assemble_generated_runtime(
        capture,
        output,
        spec=spec,
        model_sha256=MODEL_SHA256,
    )

    assert runtime.root == output / MODEL_SHA256
    assert runtime.split4_mask.read_bytes()[8] == 0x05
    assert runtime.split5_mask.read_bytes() == bytes([0xAA] * 1292)
    assert len(list((runtime.root / "fastconv").glob("*.vkfc"))) == 55
    assert not list(output.glob(".staging-*"))


def test_assemble_generated_runtime_refuses_existing_destination(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture"
    _write_raw_capture(capture)
    output = tmp_path / "generated"
    spec = load_runtime_spec(Path("config/cb62-ades-runtime.json"))
    assemble_generated_runtime(
        capture, output, spec=spec, model_sha256=MODEL_SHA256
    )

    with pytest.raises(GeneratedRuntimeError, match="already exists"):
        assemble_generated_runtime(
            capture, output, spec=spec, model_sha256=MODEL_SHA256
        )


def test_generate_owner_runtime_prepares_captures_and_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "owner-model.bin"
    model.write_bytes(b"owner supplied model")
    model_sha256 = _sha256(model.read_bytes())
    spec = AdesRuntimeSpec(
        model_sha256=model_sha256,
        pipeline_evidence_sha256="0" * 64,
        image="example.invalid/ades@sha256:" + "a" * 64,
        platform="linux/amd64",
        container_user="1000:1000",
        toolchain_env="/opt/example/env.sh",
        libvasamif="/opt/example/libvasamif.so",
        cavalry_version="2.1.7",
        cavalry_hash="c5db5f1",
        dvi_sha256=MappingProxyType({}),
    )
    sources = []
    for name in ("executor.cpp", "kernel.cpp", "operator.cpp"):
        path = tmp_path / name
        path.write_text("// synthetic test source\n")
        sources.append(path)
    observed: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            observed["init"] = kwargs
            self.workspace = Path(str(kwargs["workspace"]))

        def prepare(self) -> None:
            observed["prepared"] = True

        def capture_runtime_assets(self, **kwargs: object) -> Path:
            observed["capture"] = kwargs
            capture = self.workspace / "capture"
            _write_raw_capture(capture)
            return capture

    monkeypatch.setattr(
        "verkeye.runtime.generation.DockerAdesRuntime", FakeRuntime
    )
    workspace = tmp_path / "workspace"

    runtime = generate_owner_runtime(
        model=model,
        output_base=tmp_path / "generated",
        spec=spec,
        executor_source=sources[0],
        kernel_capture_source=sources[1],
        operator_capture_source=sources[2],
        workspace=workspace,
    )

    assert observed["prepared"] is True
    assert runtime.model_sha256 == model_sha256
    assert runtime.root == tmp_path / "generated" / model_sha256
    assert workspace.is_dir()


def test_generate_owner_runtime_refuses_nonempty_explicit_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "owner-data.txt").write_text("preserve me")
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    spec = AdesRuntimeSpec(
        model_sha256=_sha256(b"model"),
        pipeline_evidence_sha256="0" * 64,
        image="example.invalid/ades@sha256:" + "a" * 64,
        platform="linux/amd64",
        container_user="1000:1000",
        toolchain_env="/opt/example/env.sh",
        libvasamif="/opt/example/libvasamif.so",
        cavalry_version="2.1.7",
        cavalry_hash="c5db5f1",
        dvi_sha256=MappingProxyType({}),
    )

    with pytest.raises(GeneratedRuntimeError, match="workspace must be empty"):
        generate_owner_runtime(
            model=model,
            output_base=tmp_path / "generated",
            spec=spec,
            executor_source=tmp_path / "executor.cpp",
            kernel_capture_source=tmp_path / "kernel.cpp",
            operator_capture_source=tmp_path / "operator.cpp",
            workspace=workspace,
        )

    assert (workspace / "owner-data.txt").read_text() == "preserve me"
