"""Digest-pinned Docker orchestration for exact CB62 ADES execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .ades import (
    AdesExecutionResult,
    CavalryManifest,
    assemble_cb62_predictions,
    parse_cavalry_verbose,
    parse_execution_result,
    read_execution_tensors,
    render_native_manifest,
)


class AdesRuntimeError(RuntimeError):
    """Raised when a pinned runtime artifact or execution gate fails."""


_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_INPUT_BYTES = 1 * 3 * 608 * 1088


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AdesRuntimeSpec:
    model_sha256: str
    pipeline_evidence_sha256: str
    image: str
    platform: str
    container_user: str
    toolchain_env: str
    libvasamif: str
    cavalry_version: str
    cavalry_hash: str
    dvi_sha256: Mapping[str, str]

    def verify_model(self, path: str | Path) -> str:
        model = Path(path)
        try:
            observed = _sha256(model)
        except OSError as error:
            raise AdesRuntimeError(f"cannot read model {model}") from error
        if observed != self.model_sha256:
            raise AdesRuntimeError(
                "model SHA-256 mismatch: "
                f"expected {self.model_sha256}, observed {observed}"
            )
        return observed


@dataclass(frozen=True, slots=True)
class AdesPreparedRuntime:
    workspace: Path
    manifest: CavalryManifest
    dvi_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class AdesInference:
    run_id: str
    output_directory: Path
    execution: AdesExecutionResult
    tensors: Mapping[str, NDArray[np.float32]]
    predictions: NDArray[np.float32]
    tensor_sha256: Mapping[str, str]
    wall_ms: int
    timings_ns: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for name, value in self.timings_ns.items()
        ):
            raise ValueError("ADES timings must be named non-negative integers")
        object.__setattr__(
            self,
            "timings_ns",
            MappingProxyType(dict(sorted(self.timings_ns.items()))),
        )


def load_runtime_spec(path: str | Path) -> AdesRuntimeSpec:
    """Load and strictly validate the immutable ADES runtime contract."""

    try:
        document = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AdesRuntimeError(f"cannot load ADES runtime spec {path}") from error
    if document.get("schema") != "verkeye.ades-runtime.v1":
        raise AdesRuntimeError("unsupported ADES runtime spec schema")
    scalar_names = (
        "model_sha256",
        "pipeline_evidence_sha256",
        "image",
        "platform",
        "container_user",
        "toolchain_env",
        "libvasamif",
        "cavalry_version",
        "cavalry_hash",
    )
    if any(not isinstance(document.get(name), str) for name in scalar_names):
        raise AdesRuntimeError("ADES runtime spec has missing scalar fields")
    if not _SHA256.fullmatch(document["model_sha256"]):
        raise AdesRuntimeError("invalid model SHA-256 in ADES runtime spec")
    if not _SHA256.fullmatch(document["pipeline_evidence_sha256"]):
        raise AdesRuntimeError("invalid pipeline evidence SHA-256 in runtime spec")
    if "@sha256:" not in document["image"]:
        raise AdesRuntimeError("ADES image must be pinned by digest")
    if document["platform"] != "linux/amd64":
        raise AdesRuntimeError("unsupported ADES container platform")
    if not re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", document["container_user"]):
        raise AdesRuntimeError("invalid ADES container user")
    raw_dvi = document.get("dvi_sha256")
    if not isinstance(raw_dvi, dict) or len(raw_dvi) != 10:
        raise AdesRuntimeError("ADES runtime spec must pin exactly ten DVI files")
    dvi: dict[str, str] = {}
    for name, digest in raw_dvi.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".dvi")
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise AdesRuntimeError("invalid DVI identity in ADES runtime spec")
        dvi[name] = digest
    return AdesRuntimeSpec(
        model_sha256=document["model_sha256"],
        pipeline_evidence_sha256=document["pipeline_evidence_sha256"],
        image=document["image"],
        platform=document["platform"],
        container_user=document["container_user"],
        toolchain_env=document["toolchain_env"],
        libvasamif=document["libvasamif"],
        cavalry_version=document["cavalry_version"],
        cavalry_hash=document["cavalry_hash"],
        dvi_sha256=MappingProxyType(dict(sorted(dvi.items()))),
    )


def _docker_prefix(
    spec: AdesRuntimeSpec,
    workspace: Path,
    docker_command: Sequence[str],
) -> tuple[str, ...]:
    if not docker_command or any(not item for item in docker_command):
        raise AdesRuntimeError("docker command must not be empty")
    return (
        *docker_command,
        "run",
        "--rm",
        "--platform",
        spec.platform,
        "--volume",
        f"{workspace.resolve()}:/work",
        "--user",
        spec.container_user,
        "--workdir",
        "/work",
        spec.image,
        "bash",
        "-lc",
    )


def build_prepare_argv(
    spec: AdesRuntimeSpec,
    workspace: str | Path,
    *,
    docker_command: Sequence[str] = ("docker",),
) -> tuple[str, ...]:
    """Build the shell-free host invocation for deterministic preparation."""

    command = (
        "set +u; set -o pipefail; "
        f"source {spec.toolchain_env}; env_status=$?; "
        "if [ \"$env_status\" -ne 0 ] && [ \"$env_status\" -ne 1 ]; then "
        "exit \"$env_status\"; fi; set -eu; "
        f"set +e; cavalry_gen -V {spec.cavalry_version} -f /work/model.bin "
        "-p /work -v > /work/cavalry-verbose.txt "
        "2> /work/cavalry-stderr.log; cavalry_status=$?; set -e; "
        "if [ \"$cavalry_status\" -ne 0 ] && "
        "[ \"$cavalry_status\" -ne 1 ]; then "
        "exit \"$cavalry_status\"; fi; "
        "g++ -std=c++17 -O2 -Wall -Wextra /work/ades_executor.cpp "
        "-ldl -o /work/ades-executor"
    )
    return (
        *_docker_prefix(spec, Path(workspace), docker_command),
        command,
    )


def build_run_argv(
    spec: AdesRuntimeSpec,
    workspace: str | Path,
    run_id: str,
    *,
    docker_command: Sequence[str] = ("docker",),
) -> tuple[str, ...]:
    """Build one bounded exact-model inference invocation."""

    if not _RUN_ID.fullmatch(run_id):
        raise AdesRuntimeError("invalid ADES run id")
    root = f"/work/{run_id}"
    command = (
        "set +u; set -o pipefail; "
        f"source {spec.toolchain_env}; env_status=$?; "
        "if [ \"$env_status\" -ne 0 ] && [ \"$env_status\" -ne 1 ]; then "
        "exit \"$env_status\"; fi; set -eu; "
        f"/work/ades-executor {spec.libvasamif} /work/manifest.tsv "
        f"/work/parse {root}/input.tensor {root} "
        f"> {root}/stdout.tsv 2> {root}/stderr.log"
    )
    return (
        *_docker_prefix(spec, Path(workspace), docker_command),
        command,
    )


def build_capture_argv(
    spec: AdesRuntimeSpec,
    workspace: str | Path,
    *,
    docker_command: Sequence[str] = ("docker",),
) -> tuple[str, ...]:
    """Build the bounded owner-local parameter and static-mask capture."""

    command = (
        "set +u; set -o pipefail; "
        f"source {spec.toolchain_env}; env_status=$?; "
        "if [ \"$env_status\" -ne 0 ] && [ \"$env_status\" -ne 1 ]; then "
        "exit \"$env_status\"; fi; set -eu; "
        "if [ -e /work/capture ]; then "
        "echo 'capture workspace already exists' >&2; exit 73; fi; "
        "mkdir -p /work/capture/fastconv /work/capture/operators "
        "/work/capture/masks; "
        "g++ -std=c++17 -shared -fPIC -O2 "
        "-o /work/ades_kernel_capture.so /work/ades_kernel_capture.cpp -ldl; "
        "g++ -std=c++17 -shared -fPIC -O2 "
        "-o /work/ades_operator_capture.so /work/ades_operator_capture.cpp -ldl; "
        f"head -c {_INPUT_BYTES} /dev/zero > /work/capture/input.tensor; "
        "env VERKEYE_FASTCONV_DUMP=/work/capture/fastconv "
        "VERKEYE_OPERATOR_DUMP=/work/capture/operators "
        "VERKEYE_MASK_DUMP=/work/capture/masks "
        "LD_PRELOAD=/work/ades_kernel_capture.so:/work/ades_operator_capture.so "
        f"/work/ades-executor {spec.libvasamif} /work/manifest.tsv "
        "/work/parse /work/capture/input.tensor /work/capture/run "
        "> /work/capture/stdout.tsv 2> /work/capture/stderr.log"
    )
    return (
        *_docker_prefix(spec, Path(workspace), docker_command),
        command,
    )


Runner = Callable[..., subprocess.CompletedProcess[str]]


class DockerAdesRuntime:
    """Prepare and execute the exact recovered model in the pinned toolchain."""

    def __init__(
        self,
        *,
        spec: AdesRuntimeSpec,
        model: str | Path,
        workspace: str | Path,
        executor_source: str | Path,
        docker_command: Sequence[str] = ("docker",),
        runner: Runner = subprocess.run,
    ) -> None:
        self.spec = spec
        self.model = Path(model)
        self.workspace = Path(workspace)
        self.executor_source = Path(executor_source)
        self.docker_command = tuple(docker_command)
        self._runner = runner

    def _run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = self._runner(
                tuple(argv),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise AdesRuntimeError(f"cannot execute {argv[0]!r}") from error
        if result.returncode != 0:
            command = " ".join(argv[:3])
            raise AdesRuntimeError(
                f"ADES command failed ({result.returncode}): {command}; "
                f"stderr={result.stderr.strip()}"
            )
        return result

    def _verify_container_image(self) -> None:
        inspect_argv = (
            *self.docker_command,
            "image",
            "inspect",
            "--format",
            "{{.Os}}/{{.Architecture}}",
            self.spec.image,
        )
        try:
            result = self._run(inspect_argv)
        except AdesRuntimeError:
            self._run(
                (
                    *self.docker_command,
                    "pull",
                    "--platform",
                    self.spec.platform,
                    self.spec.image,
                )
            )
            result = self._run(inspect_argv)
        if result.stdout.strip() != self.spec.platform:
            raise AdesRuntimeError(
                "container architecture mismatch: "
                f"expected {self.spec.platform}, observed {result.stdout.strip()}"
            )

    def prepare(self) -> AdesPreparedRuntime:
        """Generate and verify DVI artifacts and compile the native executor."""

        self.spec.verify_model(self.model)
        if not self.executor_source.is_file():
            raise AdesRuntimeError(
                f"missing native ADES executor source {self.executor_source}"
            )
        self.workspace.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.model, self.workspace / "model.bin")
        shutil.copyfile(
            self.executor_source,
            self.workspace / "ades_executor.cpp",
        )
        parse_directory = self.workspace / "parse"
        if parse_directory.exists():
            shutil.rmtree(parse_directory)
        self._verify_container_image()
        self._run(
            build_prepare_argv(
                self.spec,
                self.workspace,
                docker_command=self.docker_command,
            )
        )
        verbose_path = self.workspace / "cavalry-verbose.txt"
        try:
            manifest = parse_cavalry_verbose(verbose_path.read_text())
        except (OSError, ValueError) as error:
            raise AdesRuntimeError("invalid Cavalry preparation output") from error
        if (
            manifest.version != self.spec.cavalry_version
            or manifest.build_hash.lower() != self.spec.cavalry_hash.lower()
        ):
            raise AdesRuntimeError("prepared Cavalry identity differs from spec")
        observed = self._verify_dvi()
        (self.workspace / "manifest.tsv").write_text(
            render_native_manifest(manifest)
        )
        if not (self.workspace / "ades-executor").is_file():
            raise AdesRuntimeError("native ADES executor was not produced")
        record = {
            "schema": "verkeye.ades-preparation.v1",
            "model_sha256": self.spec.model_sha256,
            "image": self.spec.image,
            "platform": self.spec.platform,
            "cavalry_version": manifest.version,
            "cavalry_hash": manifest.build_hash,
            "dvi_sha256": dict(observed),
            "executor_sha256": _sha256(self.workspace / "ades-executor"),
        }
        (self.workspace / "preparation.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        return AdesPreparedRuntime(
            workspace=self.workspace,
            manifest=manifest,
            dvi_sha256=observed,
        )

    def _verify_dvi(self) -> Mapping[str, str]:
        paths = sorted((self.workspace / "parse").glob("*.dvi"))
        observed = {path.name: _sha256(path) for path in paths}
        expected = dict(self.spec.dvi_sha256)
        if observed != expected:
            raise AdesRuntimeError(
                "generated DVI hashes differ from the pinned exact-model contract"
            )
        return MappingProxyType(observed)

    def prepared(self) -> AdesPreparedRuntime:
        """Verify and open an already-prepared runtime without rebuilding it."""

        self.spec.verify_model(self.workspace / "model.bin")
        if not (self.workspace / "ades-executor").is_file():
            raise AdesRuntimeError("native ADES executor is not prepared")
        try:
            manifest = parse_cavalry_verbose(
                (self.workspace / "cavalry-verbose.txt").read_text()
            )
        except (OSError, ValueError) as error:
            raise AdesRuntimeError("prepared Cavalry evidence is invalid") from error
        if (
            manifest.version != self.spec.cavalry_version
            or manifest.build_hash.lower() != self.spec.cavalry_hash.lower()
        ):
            raise AdesRuntimeError("prepared Cavalry identity differs from spec")
        return AdesPreparedRuntime(
            workspace=self.workspace,
            manifest=manifest,
            dvi_sha256=self._verify_dvi(),
        )

    def capture_runtime_assets(
        self,
        *,
        kernel_capture_source: str | Path,
        operator_capture_source: str | Path,
    ) -> Path:
        """Capture model-derived parameters and masks in the prepared runtime."""

        self.prepared()
        sources = (
            (Path(kernel_capture_source), self.workspace / "ades_kernel_capture.cpp"),
            (
                Path(operator_capture_source),
                self.workspace / "ades_operator_capture.cpp",
            ),
        )
        for source, destination in sources:
            if not source.is_file():
                raise AdesRuntimeError(f"missing native capture source {source}")
            shutil.copyfile(source, destination)
        self._verify_container_image()
        self._run(
            build_capture_argv(
                self.spec,
                self.workspace,
                docker_command=self.docker_command,
            )
        )
        capture = self.workspace / "capture"
        if not (capture / "stdout.tsv").is_file():
            raise AdesRuntimeError("ADES runtime capture did not produce evidence")
        return capture

    def infer(
        self,
        input_tensor: bytes | bytearray | memoryview,
        *,
        run_id: str,
    ) -> AdesInference:
        """Execute all ten splits and return verified raw tensors/predictions."""

        if not _RUN_ID.fullmatch(run_id):
            raise AdesRuntimeError("invalid ADES run id")
        payload = bytes(input_tensor)
        if len(payload) != _INPUT_BYTES:
            raise AdesRuntimeError(
                f"CB62 input tensor must be exactly {_INPUT_BYTES} bytes"
            )
        backend_started = time.monotonic_ns()
        phase_started = backend_started
        prepared = self.prepared()
        prepared_verify_ns = time.monotonic_ns() - phase_started
        run_directory = self.workspace / run_id
        if run_directory.exists():
            raise AdesRuntimeError(f"ADES run directory already exists: {run_id}")
        run_directory.mkdir()
        phase_started = time.monotonic_ns()
        (run_directory / "input.tensor").write_bytes(payload)
        input_write_ns = time.monotonic_ns() - phase_started
        started = time.monotonic_ns()
        try:
            self._run(
                build_run_argv(
                    self.spec,
                    self.workspace,
                    run_id,
                    docker_command=self.docker_command,
                )
            )
        finally:
            container_total_ns = time.monotonic_ns() - started
            wall_ms = container_total_ns // 1_000_000
        phase_started = time.monotonic_ns()
        try:
            execution = parse_execution_result(
                (run_directory / "stdout.tsv").read_text()
            )
        except (OSError, ValueError) as error:
            raise AdesRuntimeError("invalid ADES execution output") from error
        execution_parse_ns = time.monotonic_ns() - phase_started
        phase_started = time.monotonic_ns()
        tensors = read_execution_tensors(
            execution,
            run_directory,
            prepared.manifest,
        )
        tensor_read_ns = time.monotonic_ns() - phase_started
        phase_started = time.monotonic_ns()
        predictions = assemble_cb62_predictions(tensors)
        prediction_assembly_ns = time.monotonic_ns() - phase_started
        phase_started = time.monotonic_ns()
        hashes: dict[str, str] = {}
        for split in execution.splits:
            for tensor in split.tensors:
                path = run_directory / Path(tensor.path).name
                hashes[path.name] = _sha256(path)
        tensor_hash_ns = time.monotonic_ns() - phase_started
        (run_directory / "wall-ms.txt").write_text(f"{wall_ms}\n")
        backend_total_ns = time.monotonic_ns() - backend_started
        split_execution_ns = sum(
            split.duration_us for split in execution.splits
        ) * 1_000
        timings_ns = {
            "backend_total": backend_total_ns,
            "prepared_verify": prepared_verify_ns,
            "input_write": input_write_ns,
            "container_total": container_total_ns,
            "split_execution": split_execution_ns,
            "container_overhead": max(0, container_total_ns - split_execution_ns),
            "execution_parse": execution_parse_ns,
            "tensor_read": tensor_read_ns,
            "prediction_assembly": prediction_assembly_ns,
            "tensor_hash": tensor_hash_ns,
        }
        return AdesInference(
            run_id=run_id,
            output_directory=run_directory,
            execution=execution,
            tensors=tensors,
            predictions=predictions,
            tensor_sha256=MappingProxyType(dict(sorted(hashes.items()))),
            wall_ms=wall_ms,
            timings_ns=timings_ns,
        )
