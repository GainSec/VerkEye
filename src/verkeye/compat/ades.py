"""Strict parsing and memory planning for Ambarella ADES host execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


class CavalryManifestError(ValueError):
    """Raised when Cavalry metadata is incomplete or internally inconsistent."""


class AdesExecutionError(ValueError):
    """Raised when native ADES execution evidence is incomplete or invalid."""


@dataclass(frozen=True)
class CavalryPort:
    direction: str
    index: int
    name: str
    size: int
    byte_offset: int
    shape: tuple[int, int, int, int]
    pitch: int
    pitch_offset: int
    pitch_bits: int
    dram_format: int
    data_format: tuple[int, int, int, int]
    main_input_output: bool


@dataclass(frozen=True)
class CavalrySplit:
    index: int
    name: str
    image_start: int
    image_size: int
    dag_start: int
    inputs: tuple[CavalryPort, ...]
    outputs: tuple[CavalryPort, ...]

    @property
    def dag_offset(self) -> int:
        return self.dag_start - self.image_start


@dataclass(frozen=True)
class CavalryManifest:
    version: str
    build_hash: str
    splits: tuple[CavalrySplit, ...]


@dataclass(frozen=True)
class AdesTensorRecord:
    split_index: int
    output_index: int
    name: str
    path: str
    size: int
    changed_bytes: int
    main_input_output: bool


@dataclass(frozen=True)
class AdesSplitResult:
    index: int
    duration_us: int
    tensors: tuple[AdesTensorRecord, ...]


@dataclass(frozen=True)
class AdesExecutionResult:
    cavalry_version: str
    cavalry_hash: str
    splits: tuple[AdesSplitResult, ...]


_VERSION = re.compile(
    r"^version\s+(?P<version>\S+)\s+\( HASH (?P<hash>[0-9a-fA-F]+) \)"
    r"\s+dvi_num:\s*(?P<count>\d+)\s*$"
)
_SUMMARY = re.compile(
    r"^dvi_id:\s*(?P<id>\d+)\s+"
    r"dvi_img_vaddr:\s*(?P<image_start>\d+)\s+"
    r"dvi_img_size:\s*(?P<image_size>\d+)\s+"
    r"dvi_dag_vaddr:\s*(?P<dag_start>\d+)\s+"
    r"input_num:\s*(?P<inputs>\d+)\s+"
    r"output_num:\s*(?P<outputs>\d+)\s+.*?"
    r"dag_name:\s*(?P<name>\S+)\s*$"
)
_DETAIL = re.compile(
    r"^dvi_id:\s*(?P<id>\d+)\s+dag_name:\s*(?P<name>\S+)\s+"
    r"input_num:\s*(?P<inputs>\d+)\s+output_num:\s*(?P<outputs>\d+)\s*$"
)
_PORT = re.compile(r"^(?P<direction>input|output)_id:\s*(?P<id>\d+)\b")


def _integer(line: str, label: str) -> int:
    match = re.search(rf"\b{re.escape(label)}\s*:\s*(\d+)", line)
    if not match:
        raise CavalryManifestError(f"missing {label!r} in port record")
    return int(match.group(1))


def _tuple(line: str, label: str, count: int) -> tuple[int, ...]:
    match = re.search(
        rf"\b{re.escape(label)}\s*:\s*\(([^)]*)\)", line
    )
    if not match:
        raise CavalryManifestError(f"missing {label!r} in port record")
    try:
        values = tuple(int(item.strip()) for item in match.group(1).split(","))
    except ValueError as error:
        raise CavalryManifestError(f"invalid {label!r} tuple") from error
    if len(values) != count:
        raise CavalryManifestError(
            f"{label!r} must contain {count} values, found {len(values)}"
        )
    return values


def _shape(line: str) -> tuple[int, int, int, int]:
    match = re.search(
        r"\bdim\s*:\s*\(P, D, H, W\)\s*=\s*\(([^)]*)\)", line
    )
    if not match:
        raise CavalryManifestError("missing 'dim' in port record")
    try:
        values = tuple(int(item.strip()) for item in match.group(1).split(","))
    except ValueError as error:
        raise CavalryManifestError("invalid 'dim' tuple") from error
    if len(values) != 4:
        raise CavalryManifestError(
            f"'dim' must contain 4 values, found {len(values)}"
        )
    return values[0], values[1], values[2], values[3]


def _parse_port(line: str) -> CavalryPort:
    match = _PORT.match(line)
    if not match:
        raise CavalryManifestError("invalid port record")
    name_match = re.search(r"\bport_name:\s*(.*?)\s+\(layer_name:", line)
    if not name_match or not name_match.group(1):
        raise CavalryManifestError("missing port name")
    shape = _shape(line)
    data_format = _tuple(line, "data_format", 4)
    return CavalryPort(
        direction=match.group("direction"),
        index=int(match.group("id")),
        name=name_match.group(1),
        size=_integer(line, "port_size"),
        byte_offset=_integer(line, "port_byte_offset"),
        shape=(shape[0], shape[1], shape[2], shape[3]),
        pitch=_integer(line, "pitch"),
        pitch_offset=_integer(line, "pitch_offset"),
        pitch_bits=_integer(line, "pitch_bsize"),
        dram_format=_integer(line, "dram_format"),
        data_format=(
            data_format[0], data_format[1], data_format[2], data_format[3]
        ),
        main_input_output=bool(_integer(line, "is_main_input_output")),
    )


def parse_cavalry_verbose(text: str) -> CavalryManifest:
    """Parse ``cavalry_gen -v`` output and reject incomplete graph metadata."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    version_match = next((_VERSION.match(line) for line in lines if _VERSION.match(line)), None)
    if version_match is None:
        raise CavalryManifestError("missing Cavalry version header")
    expected_splits = int(version_match.group("count"))

    summaries: dict[int, dict[str, int | str]] = {}
    for line in lines:
        match = _SUMMARY.match(line)
        if not match:
            continue
        index = int(match.group("id"))
        if index in summaries:
            raise CavalryManifestError(f"duplicate split summary {index}")
        summaries[index] = {
            "name": match.group("name"),
            "image_start": int(match.group("image_start")),
            "image_size": int(match.group("image_size")),
            "dag_start": int(match.group("dag_start")),
            "inputs": int(match.group("inputs")),
            "outputs": int(match.group("outputs")),
        }
    if len(summaries) != expected_splits:
        raise CavalryManifestError(
            f"expected {expected_splits} split summaries, found {len(summaries)}"
        )

    details: dict[int, dict[str, object]] = {}
    active: int | None = None
    for line in lines:
        detail = _DETAIL.match(line)
        if detail:
            active = int(detail.group("id"))
            if active in details:
                raise CavalryManifestError(f"duplicate split detail {active}")
            details[active] = {
                "name": detail.group("name"),
                "inputs": int(detail.group("inputs")),
                "outputs": int(detail.group("outputs")),
                "ports": [],
            }
            continue
        if active is not None and _PORT.match(line):
            ports = details[active]["ports"]
            assert isinstance(ports, list)
            ports.append(_parse_port(line))

    if set(details) != set(summaries):
        raise CavalryManifestError("split detail set does not match summaries")

    splits: list[CavalrySplit] = []
    for index in range(expected_splits):
        if index not in summaries:
            raise CavalryManifestError(f"missing split {index}")
        summary = summaries[index]
        detail = details[index]
        if summary["name"] != detail["name"]:
            raise CavalryManifestError(f"split {index} name mismatch")
        ports = detail["ports"]
        assert isinstance(ports, list)
        inputs = tuple(port for port in ports if port.direction == "input")
        outputs = tuple(port for port in ports if port.direction == "output")
        expected_inputs = int(summary["inputs"])
        expected_outputs = int(summary["outputs"])
        if int(detail["inputs"]) != expected_inputs or len(inputs) != expected_inputs:
            raise CavalryManifestError(f"split {index} input count mismatch")
        if int(detail["outputs"]) != expected_outputs or len(outputs) != expected_outputs:
            raise CavalryManifestError(f"split {index} output count mismatch")
        if tuple(port.index for port in inputs) != tuple(range(len(inputs))):
            raise CavalryManifestError(f"split {index} input ids are not contiguous")
        if tuple(port.index for port in outputs) != tuple(range(len(outputs))):
            raise CavalryManifestError(f"split {index} output ids are not contiguous")
        image_start = int(summary["image_start"])
        image_size = int(summary["image_size"])
        dag_start = int(summary["dag_start"])
        if dag_start < image_start or dag_start >= image_start + image_size:
            raise CavalryManifestError(f"split {index} DAG address is outside image")
        splits.append(
            CavalrySplit(
                index=index,
                name=str(summary["name"]),
                image_start=image_start,
                image_size=image_size,
                dag_start=dag_start,
                inputs=inputs,
                outputs=outputs,
            )
        )

    manifest = CavalryManifest(
        version=version_match.group("version"),
        build_hash=version_match.group("hash"),
        splits=tuple(splits),
    )
    allocate_tensor_addresses(manifest)
    return manifest


def _edge_signature(port: CavalryPort) -> tuple[object, ...]:
    return (
        port.size,
        port.shape,
        port.pitch,
        port.pitch_bits,
        port.dram_format,
        port.data_format,
    )


def allocate_tensor_addresses(
    manifest: CavalryManifest,
    *,
    start: int = 0x100000,
    alignment: int = 0x1000,
) -> dict[str, int]:
    """Assign one bounded DRAM address to every named graph edge."""

    if start < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise CavalryManifestError("address plan requires power-of-two alignment")
    known: dict[str, CavalryPort] = {}
    addresses: dict[str, int] = {}
    cursor = (start + alignment - 1) & ~(alignment - 1)
    for split in manifest.splits:
        for port in (*split.inputs, *split.outputs):
            if "\t" in port.name or "\n" in port.name:
                raise CavalryManifestError("port name is not manifest-safe")
            previous = known.get(port.name)
            if previous is not None:
                if _edge_signature(previous) != _edge_signature(port):
                    raise CavalryManifestError(
                        f"inconsistent graph edge metadata for {port.name!r}"
                    )
                continue
            known[port.name] = port
            addresses[port.name] = cursor
            cursor = (cursor + port.size + alignment - 1) & ~(alignment - 1)
    return addresses


def render_native_manifest(manifest: CavalryManifest) -> str:
    """Render a strict tab-delimited contract consumed by the native runner."""

    addresses = allocate_tensor_addresses(manifest)
    lines = [
        f"ADES_MANIFEST\t1\t{manifest.version}\t{manifest.build_hash}",
    ]
    for split in manifest.splits:
        lines.append(
            "\t".join(
                (
                    "S",
                    str(split.index),
                    split.name,
                    str(split.image_start),
                    str(split.image_size),
                    str(split.dag_start),
                )
            )
        )
        for port in (*split.inputs, *split.outputs):
            lines.append(
                "\t".join(
                    (
                        "I" if port.direction == "input" else "O",
                        str(split.index),
                        port.name,
                        str(port.byte_offset),
                        str(port.size),
                        str(addresses[port.name]),
                        *(str(value) for value in port.shape),
                        str(port.pitch),
                        str(port.pitch_offset),
                        str(port.pitch_bits),
                        str(port.dram_format),
                        *(str(value) for value in port.data_format),
                        "1" if port.main_input_output else "0",
                    )
                )
            )
    lines.append("END")
    return "\n".join(lines) + "\n"


def _execution_integer(value: str, label: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise AdesExecutionError(f"invalid {label}")
    return int(value)


def parse_execution_result(text: str) -> AdesExecutionResult:
    """Parse the native runner's strict, tab-delimited execution record."""

    lines = text.splitlines()
    if not lines:
        raise AdesExecutionError("missing ADES_RESULT header")
    header = lines[0].split("\t")
    if len(header) != 4 or header[:2] != ["ADES_RESULT", "1"]:
        raise AdesExecutionError("unsupported ADES_RESULT header")
    if not header[2] or not re.fullmatch(r"[0-9a-fA-F]+", header[3]):
        raise AdesExecutionError("invalid ADES_RESULT identity")

    splits: list[AdesSplitResult] = []
    active_index: int | None = None
    active_duration = 0
    active_tensors: list[AdesTensorRecord] = []
    ended = False

    def finish_active() -> None:
        nonlocal active_index, active_duration, active_tensors
        if active_index is None:
            return
        splits.append(
            AdesSplitResult(
                index=active_index,
                duration_us=active_duration,
                tensors=tuple(active_tensors),
            )
        )
        active_index = None
        active_duration = 0
        active_tensors = []

    for line in lines[1:]:
        fields = line.split("\t")
        if fields == ["END"]:
            finish_active()
            ended = True
            continue
        if ended:
            raise AdesExecutionError("unexpected record after END")
        if fields[0] == "SPLIT":
            if len(fields) != 3:
                raise AdesExecutionError("invalid SPLIT record")
            finish_active()
            active_index = _execution_integer(fields[1], "split index")
            if active_index != len(splits):
                raise AdesExecutionError("split indexes are not contiguous")
            active_duration = _execution_integer(fields[2], "split duration")
            continue
        if fields[0] == "TENSOR":
            if len(fields) != 8 or active_index is None:
                raise AdesExecutionError("invalid TENSOR record")
            split_index = _execution_integer(fields[1], "tensor split index")
            output_index = _execution_integer(fields[2], "tensor output index")
            if split_index != active_index:
                raise AdesExecutionError("tensor split does not match active split")
            if output_index != len(active_tensors):
                raise AdesExecutionError("tensor output indexes are not contiguous")
            if not fields[3] or not fields[4]:
                raise AdesExecutionError("tensor name and path must not be empty")
            if fields[7] not in {"0", "1"}:
                raise AdesExecutionError("invalid tensor main-output flag")
            size = _execution_integer(fields[5], "tensor size")
            changed = _execution_integer(fields[6], "tensor changed-byte count")
            if size == 0 or changed > size:
                raise AdesExecutionError("invalid tensor byte counts")
            active_tensors.append(
                AdesTensorRecord(
                    split_index=split_index,
                    output_index=output_index,
                    name=fields[3],
                    path=fields[4],
                    size=size,
                    changed_bytes=changed,
                    main_input_output=fields[7] == "1",
                )
            )
            continue
        raise AdesExecutionError("unknown execution record")

    if not ended:
        raise AdesExecutionError("missing END marker")
    if not splits:
        raise AdesExecutionError("execution contains no splits")
    return AdesExecutionResult(
        cavalry_version=header[2],
        cavalry_hash=header[3],
        splits=tuple(splits),
    )


def decode_float32_chw(
    payload: bytes,
    port: CavalryPort,
) -> NDArray[np.float32]:
    """Decode one pitched Cavalry main-output buffer as exact float32 CHW."""

    if len(payload) != port.size:
        raise AdesExecutionError(
            f"tensor {port.name!r} size mismatch: "
            f"expected {port.size}, observed {len(payload)}"
        )
    if (
        port.pitch_bits != 32
        or port.dram_format != 0
        or port.data_format != (1, 2, 0, 7)
    ):
        raise AdesExecutionError(
            f"tensor {port.name!r} is not a proved float32 main output"
        )
    planes, depth, height, width = port.shape
    row_bytes = width * 4
    if port.pitch < row_bytes:
        raise AdesExecutionError(f"tensor {port.name!r} pitch is too small")
    expected_size = planes * depth * height * port.pitch
    if expected_size != port.size:
        raise AdesExecutionError(
            f"tensor {port.name!r} shape/pitch size is inconsistent"
        )
    view = np.ndarray(
        shape=(planes, depth, height, width),
        dtype="<f4",
        buffer=payload,
        strides=(depth * height * port.pitch, height * port.pitch, port.pitch, 4),
    )
    result = np.ascontiguousarray(view, dtype=np.float32)
    result.setflags(write=False)
    return result


def read_execution_tensors(
    result: AdesExecutionResult,
    output_directory: str | Path,
    manifest: CavalryManifest,
) -> Mapping[str, NDArray[np.float32]]:
    """Read and decode all six exact CB62 main-output tensors."""

    if result.cavalry_version != manifest.version or result.cavalry_hash.lower() != manifest.build_hash.lower():
        raise AdesExecutionError("execution and manifest Cavalry identities differ")
    if len(result.splits) != len(manifest.splits):
        raise AdesExecutionError("execution and manifest split counts differ")
    root = Path(output_directory)
    tensors: dict[str, NDArray[np.float32]] = {}
    for split_result, split_spec in zip(result.splits, manifest.splits, strict=True):
        if split_result.index != split_spec.index:
            raise AdesExecutionError("execution split order differs from manifest")
        if len(split_result.tensors) != len(split_spec.outputs):
            raise AdesExecutionError(
                f"split {split_spec.index} output count differs from manifest"
            )
        for record, port in zip(
            split_result.tensors, split_spec.outputs, strict=True
        ):
            if record.output_index != port.index or record.name != port.name:
                raise AdesExecutionError(
                    f"split {split_spec.index} output identity differs from manifest"
                )
            if record.size != port.size or record.main_input_output != port.main_input_output:
                raise AdesExecutionError(
                    f"split {split_spec.index} output metadata differs from manifest"
                )
            if not port.main_input_output:
                continue
            path = root / Path(record.path).name
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise AdesExecutionError(f"cannot read tensor {path}") from error
            if port.name in tensors:
                raise AdesExecutionError(f"duplicate main output {port.name!r}")
            tensors[port.name] = decode_float32_chw(payload, port)
    expected = {f"output_{index}" for index in range(6)}
    if set(tensors) != expected:
        raise AdesExecutionError(
            f"main output set mismatch: expected {sorted(expected)}, "
            f"observed {sorted(tensors)}"
        )
    return MappingProxyType(tensors)


def assemble_cb62_predictions(
    tensors: Mapping[str, NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Assemble the three paired CHW heads into the exact 13,566x8 ABI."""

    pairs = (
        ("output_3", "output_0", (1, 4, 76, 136)),
        ("output_4", "output_1", (1, 4, 38, 68)),
        ("output_5", "output_2", (1, 4, 19, 34)),
    )
    rows: list[NDArray[np.float32]] = []
    for box_name, score_name, expected_shape in pairs:
        try:
            boxes = np.asarray(tensors[box_name])
            scores = np.asarray(tensors[score_name])
        except KeyError as error:
            raise AdesExecutionError(
                f"missing CB62 output tensor {error.args[0]!r}"
            ) from error
        for name, tensor in ((box_name, boxes), (score_name, scores)):
            if tensor.dtype != np.float32 or tensor.shape != expected_shape:
                raise AdesExecutionError(
                    f"tensor {name!r} must be float32{expected_shape}"
                )
            if not np.all(np.isfinite(tensor)):
                raise AdesExecutionError(f"tensor {name!r} contains non-finite values")
        paired = np.concatenate((boxes, scores), axis=1)
        rows.append(paired.transpose(0, 2, 3, 1).reshape(-1, 8))
    result = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
    if result.shape != (13_566, 8):
        raise AdesExecutionError("assembled CB62 prediction shape is invalid")
    result.setflags(write=False)
    return result
