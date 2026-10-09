"""Owner-local generation and verification of accelerated CB62 runtime assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from ..cv22.parameter_extraction import (
    FastconvCaptureError,
    encode_fastconv_capture,
    parse_fastconv_capture,
)
from ..compat.ades_runtime import DockerAdesRuntime

if TYPE_CHECKING:
    from ..compat.ades_runtime import AdesRuntimeSpec


class GeneratedRuntimeError(RuntimeError):
    """A generated runtime is absent, malformed, or fails integrity checks."""


_SHA256 = re.compile(r"[0-9a-f]{64}")
_MASK_CONTRACTS = MappingProxyType(
    {
        "split4": (38, 68, 342),
        "split5": (76, 136, 1292),
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise GeneratedRuntimeError(
            f"cannot read generated runtime member {path}"
        ) from error
    return digest.hexdigest()


def _member_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise GeneratedRuntimeError("generated runtime has an invalid member path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise GeneratedRuntimeError(
            "generated runtime requires a safe relative member path"
        )
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        raise GeneratedRuntimeError(f"generated runtime member is missing: {value}")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise GeneratedRuntimeError(
            f"generated runtime member resolves outside runtime root: {value}"
        ) from error
    return path


def _verify_member(root: Path, record: object) -> tuple[Path, int, str]:
    if not isinstance(record, dict):
        raise GeneratedRuntimeError("generated runtime member must be an object")
    path = _member_path(root, record.get("path"))
    size = record.get("size")
    digest = record.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise GeneratedRuntimeError("generated runtime member has invalid size")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise GeneratedRuntimeError("generated runtime member has invalid SHA-256")
    try:
        observed_size = path.stat().st_size
    except OSError as error:
        raise GeneratedRuntimeError(
            f"cannot stat generated runtime member {path}"
        ) from error
    if observed_size != size:
        raise GeneratedRuntimeError(
            f"generated runtime size mismatch for {path}: {observed_size} != {size}"
        )
    observed_digest = _sha256(path)
    if observed_digest != digest:
        raise GeneratedRuntimeError(f"generated runtime SHA-256 mismatch for {path}")
    return path, size, digest


@dataclass(frozen=True, slots=True)
class GeneratedFastconvMember:
    call_index: int
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GeneratedRuntime:
    root: Path
    model_sha256: str
    toolchain_image: str
    toolchain_platform: str
    fastconv: tuple[GeneratedFastconvMember, ...]
    masks: Mapping[str, Path]

    @property
    def split4_mask(self) -> Path:
        return self.masks["split4"]

    @property
    def split5_mask(self) -> Path:
        return self.masks["split5"]

    def fastconv_capture(self, call_index: int):
        """Decode one already integrity-checked normalized parameter package."""

        if not isinstance(call_index, int) or isinstance(call_index, bool):
            raise GeneratedRuntimeError("fastconv call index must be an integer")
        if not 0 <= call_index < len(self.fastconv):
            raise GeneratedRuntimeError("fastconv call index is out of range")
        from ..cv22.parameter_extraction import (
            FastconvCaptureError,
            decode_fastconv_capture,
        )

        member = self.fastconv[call_index]
        try:
            return decode_fastconv_capture(member.path.read_bytes())
        except (OSError, FastconvCaptureError) as error:
            raise GeneratedRuntimeError(
                f"cannot decode generated fastconv call {call_index}"
            ) from error


def generated_runtime_path(base: str | Path, model_sha256: str) -> Path:
    """Return the content-addressed destination for one supported model."""

    if not _SHA256.fullmatch(model_sha256):
        raise GeneratedRuntimeError("invalid model SHA-256")
    return Path(base) / model_sha256


def load_generated_runtime(
    root: str | Path,
    *,
    expected_model_sha256: str | None = None,
) -> GeneratedRuntime:
    """Load a complete generated runtime and verify every declared byte."""

    runtime_root = Path(root)
    manifest_path = runtime_root / "manifest.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeneratedRuntimeError(
            f"cannot load generated runtime manifest {manifest_path}"
        ) from error
    if (
        not isinstance(document, dict)
        or document.get("schema") != "verkeye.generated-runtime.v1"
    ):
        raise GeneratedRuntimeError("unsupported generated runtime manifest schema")
    model_sha256 = document.get("model_sha256")
    if not isinstance(model_sha256, str) or not _SHA256.fullmatch(model_sha256):
        raise GeneratedRuntimeError("generated runtime has invalid model SHA-256")
    if expected_model_sha256 is not None and model_sha256 != expected_model_sha256:
        raise GeneratedRuntimeError(
            "generated runtime model SHA-256 mismatch: "
            f"expected {expected_model_sha256}, observed {model_sha256}"
        )
    toolchain = document.get("toolchain")
    if not isinstance(toolchain, dict):
        raise GeneratedRuntimeError("generated runtime lacks toolchain identity")
    image = toolchain.get("image")
    platform = toolchain.get("platform")
    if not isinstance(image, str) or "@sha256:" not in image:
        raise GeneratedRuntimeError(
            "generated runtime toolchain image is not digest-pinned"
        )
    if platform != "linux/amd64":
        raise GeneratedRuntimeError(
            "generated runtime has unsupported toolchain platform"
        )

    records = document.get("fastconv")
    if not isinstance(records, list) or len(records) != 55:
        raise GeneratedRuntimeError(
            "generated runtime must contain exactly 55 fastconv members"
        )
    fastconv: list[GeneratedFastconvMember] = []
    for expected_call, record in enumerate(records):
        if not isinstance(record, dict) or record.get("call_index") != expected_call:
            raise GeneratedRuntimeError(
                "generated runtime fastconv calls are not contiguous"
            )
        path, size, digest = _verify_member(runtime_root, record)
        if path.suffix != ".vkfc":
            raise GeneratedRuntimeError(
                "generated runtime fastconv member must use .vkfc"
            )
        fastconv.append(
            GeneratedFastconvMember(expected_call, path, size, digest)
        )

    mask_records = document.get("masks")
    if not isinstance(mask_records, dict) or set(mask_records) != set(_MASK_CONTRACTS):
        raise GeneratedRuntimeError(
            "generated runtime must contain split4 and split5 masks"
        )
    masks: dict[str, Path] = {}
    for name, (height, width, size) in _MASK_CONTRACTS.items():
        record = mask_records[name]
        if not isinstance(record, dict) or (
            record.get("height"), record.get("width"), record.get("size")
        ) != (height, width, size):
            raise GeneratedRuntimeError(
                f"generated runtime {name} mask contract mismatch"
            )
        path, _, _ = _verify_member(runtime_root, record)
        masks[name] = path

    return GeneratedRuntime(
        root=runtime_root,
        model_sha256=model_sha256,
        toolchain_image=image,
        toolchain_platform=platform,
        fastconv=tuple(fastconv),
        masks=MappingProxyType(masks),
    )


def _normalize_mask(payload: bytes, *, height: int, width: int) -> bytes:
    row_bytes = (width + 7) // 8
    expected = height * row_bytes
    if len(payload) != expected:
        raise GeneratedRuntimeError(
            f"captured mask size mismatch: {len(payload)} != {expected}"
        )
    normalized = bytearray(payload)
    remainder = width % 8
    if remainder:
        padding_mask = (1 << remainder) - 1
        for row in range(height):
            normalized[(row + 1) * row_bytes - 1] &= padding_mask
    return bytes(normalized)


def _raw_fastconv_capture(root: Path, call: int):
    entries_path = root / f"fastconv-{call}-entries.bin"
    try:
        entries = entries_path.read_bytes()
    except OSError as error:
        raise GeneratedRuntimeError(
            f"captured fastconv call {call} is missing"
        ) from error
    channel_count = len(entries) // 96 if entries else 0
    payloads: dict[int, bytes] = {}
    for index in range(channel_count):
        path = root / f"fastconv-{call}-entry-{index}-offset-0.bin"
        if path.is_file():
            payloads[index] = path.read_bytes()
    try:
        return parse_fastconv_capture(entries, payloads)
    except FastconvCaptureError as error:
        raise GeneratedRuntimeError(
            f"captured fastconv call {call} is invalid: {error}"
        ) from error


def _write_member(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def assemble_generated_runtime(
    capture_root: str | Path,
    output_base: str | Path,
    *,
    spec: "AdesRuntimeSpec",
    model_sha256: str,
    force: bool = False,
) -> GeneratedRuntime:
    """Normalize a complete raw ADES capture and install it atomically."""

    if model_sha256 != spec.model_sha256:
        raise GeneratedRuntimeError("model SHA-256 mismatch with runtime spec")
    capture = Path(capture_root)
    fastconv_root = capture / "fastconv"
    mask_root = capture / "masks"
    if not fastconv_root.is_dir() or not mask_root.is_dir():
        raise GeneratedRuntimeError("capture lacks fastconv or mask directories")
    output = Path(output_base)
    output.mkdir(parents=True, exist_ok=True)
    destination = generated_runtime_path(output, model_sha256)
    if destination.exists() and not force:
        raise GeneratedRuntimeError(f"generated runtime already exists: {destination}")

    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=output))
    try:
        fastconv_records: list[dict[str, object]] = []
        for call in range(55):
            encoded = encode_fastconv_capture(
                _raw_fastconv_capture(fastconv_root, call)
            )
            relative = Path("fastconv") / f"call-{call:03d}.vkfc"
            record = _write_member(staging / relative, encoded)
            record["path"] = relative.as_posix()
            record["call_index"] = call
            fastconv_records.append(record)
        extra = fastconv_root / "fastconv-55-entries.bin"
        if extra.exists():
            raise GeneratedRuntimeError("capture contains unexpected fastconv call 55")

        mask_records: dict[str, dict[str, object]] = {}
        for name, (height, width, size) in _MASK_CONTRACTS.items():
            candidates = sorted(
                path
                for path in mask_root.iterdir()
                if path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == size
            )
            if len(candidates) != 1:
                raise GeneratedRuntimeError(
                    f"capture must contain exactly one {name} mask candidate; "
                    f"found {len(candidates)}"
                )
            payload = _normalize_mask(
                candidates[0].read_bytes(), height=height, width=width
            )
            relative = Path("masks") / f"{name}.bin"
            record = _write_member(staging / relative, payload)
            record.update(
                {
                    "path": relative.as_posix(),
                    "height": height,
                    "width": width,
                }
            )
            mask_records[name] = record

        manifest = {
            "schema": "verkeye.generated-runtime.v1",
            "model_sha256": model_sha256,
            "toolchain": {
                "image": spec.image,
                "platform": spec.platform,
                "cavalry_version": spec.cavalry_version,
                "cavalry_hash": spec.cavalry_hash,
            },
            "fastconv": fastconv_records,
            "masks": mask_records,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        load_generated_runtime(staging, expected_model_sha256=model_sha256)
        if destination.exists():
            if not force:
                raise GeneratedRuntimeError(
                    f"generated runtime already exists: {destination}"
                )
            shutil.rmtree(destination)
        os.replace(staging, destination)
        return load_generated_runtime(
            destination, expected_model_sha256=model_sha256
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def generate_owner_runtime(
    *,
    model: str | Path,
    output_base: str | Path,
    spec: "AdesRuntimeSpec",
    executor_source: str | Path,
    kernel_capture_source: str | Path,
    operator_capture_source: str | Path,
    workspace: str | Path | None = None,
    docker_command: tuple[str, ...] = ("docker",),
    force: bool = False,
    keep_workspace: bool = False,
) -> GeneratedRuntime:
    """Generate a complete local runtime from one supported owner model."""

    model_path = Path(model)
    model_sha256 = spec.verify_model(model_path)
    explicit_workspace = workspace is not None
    if workspace is None:
        workspace_path = Path(
            tempfile.mkdtemp(prefix="verkeye-generation-workspace-")
        )
    else:
        workspace_path = Path(workspace)
        if workspace_path.exists():
            try:
                next(workspace_path.iterdir())
            except StopIteration:
                pass
            except OSError as error:
                raise GeneratedRuntimeError(
                    f"cannot inspect generation workspace {workspace_path}"
                ) from error
            else:
                raise GeneratedRuntimeError(
                    f"generation workspace must be empty: {workspace_path}"
                )
        else:
            workspace_path.mkdir(parents=True)
    try:
        runtime = DockerAdesRuntime(
            spec=spec,
            model=model_path,
            workspace=workspace_path,
            executor_source=executor_source,
            docker_command=docker_command,
        )
        runtime.prepare()
        capture = runtime.capture_runtime_assets(
            kernel_capture_source=kernel_capture_source,
            operator_capture_source=operator_capture_source,
        )
        return assemble_generated_runtime(
            capture,
            output_base,
            spec=spec,
            model_sha256=model_sha256,
            force=force,
        )
    finally:
        if not explicit_workspace and not keep_workspace:
            shutil.rmtree(workspace_path, ignore_errors=True)
