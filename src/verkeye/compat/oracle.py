"""Content-addressed ADES oracle bundles for differential execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_INPUT_SIZE = 1 * 3 * 608 * 1088


class OracleBundleError(ValueError):
    """An oracle bundle is incomplete, changed, or structurally invalid."""


@dataclass(frozen=True, slots=True)
class OracleBundle:
    root: Path
    case_id: str
    input_tensor: bytes
    raw_tensors: Mapping[str, bytes]
    predictions: NDArray[np.float32]
    detections: tuple[Mapping[str, Any], ...]
    model_sha256: str
    pipeline_sha256: str
    runtime_identity: Mapping[str, str]
    manifest_sha256: str


def write_oracle_bundle(
    destination: str | Path,
    *,
    case_id: str,
    input_tensor: bytes | bytearray | memoryview,
    raw_tensors: Mapping[str, bytes | bytearray | memoryview],
    predictions: NDArray[np.float32],
    detections: Sequence[Mapping[str, Any]],
    model_sha256: str,
    pipeline_sha256: str,
    runtime_identity: Mapping[str, str],
) -> Path:
    """Atomically create one immutable oracle bundle."""

    target = Path(destination)
    if target.exists():
        raise OracleBundleError(f"oracle bundle already exists: {target}")
    if not _CASE_ID.fullmatch(case_id):
        raise OracleBundleError("invalid oracle case_id")
    payload = bytes(input_tensor)
    if len(payload) != _INPUT_SIZE:
        raise OracleBundleError(f"input tensor must contain exactly {_INPUT_SIZE} bytes")
    if len(raw_tensors) != 22:
        raise OracleBundleError("oracle bundle must contain exactly 22 raw tensors")
    _require_digest(model_sha256, "model_sha256")
    _require_digest(pipeline_sha256, "pipeline_sha256")
    prediction_array = np.asarray(predictions)
    if prediction_array.dtype != np.float32 or prediction_array.shape != (13_566, 8):
        raise OracleBundleError("prediction matrix must be 13566x8 float32")
    if not np.all(np.isfinite(prediction_array)):
        raise OracleBundleError("prediction matrix contains non-finite values")
    normalized_runtime = _runtime_identity(runtime_identity)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        tensors_directory = temporary / "tensors"
        tensors_directory.mkdir()
        input_path = temporary / "input.tensor"
        prediction_path = temporary / "prediction.npy"
        detections_path = temporary / "detections.json"
        input_path.write_bytes(payload)
        with prediction_path.open("wb") as stream:
            np.save(stream, np.ascontiguousarray(prediction_array), allow_pickle=False)
        detections_document = [dict(item) for item in detections]
        detections_path.write_text(
            json.dumps(detections_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        tensor_records: list[dict[str, object]] = []
        for name, raw in sorted(raw_tensors.items()):
            if not name or Path(name).name != name:
                raise OracleBundleError(f"invalid raw tensor name: {name!r}")
            tensor_path = tensors_directory / name
            tensor_path.write_bytes(bytes(raw))
            tensor_records.append(_member_record(temporary, tensor_path, name=name))

        manifest = {
            "schema": "verkeye.ades-oracle-bundle.v1",
            "case_id": case_id,
            "model_sha256": model_sha256,
            "pipeline_sha256": pipeline_sha256,
            "runtime_identity": normalized_runtime,
            "input": _member_record(temporary, input_path),
            "prediction": {
                **_member_record(temporary, prediction_path),
                "dtype": "float32",
                "shape": [13_566, 8],
                "raw_sha256": hashlib.sha256(
                    prediction_array.tobytes(order="C")
                ).hexdigest(),
            },
            "detections": {
                **_member_record(temporary, detections_path),
                "count": len(detections_document),
            },
            "raw_tensors": tensor_records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load_oracle_bundle(path: str | Path) -> OracleBundle:
    """Strictly verify every bundle member before exposing any oracle data."""

    root = Path(path)
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        document = json.loads(manifest_bytes)
    except FileNotFoundError as exc:
        raise OracleBundleError("missing oracle manifest") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleBundleError("cannot read oracle manifest") from exc
    if document.get("schema") != "verkeye.ades-oracle-bundle.v1":
        raise OracleBundleError("unsupported oracle bundle schema")
    case_id = document.get("case_id")
    if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
        raise OracleBundleError("invalid oracle case_id")
    model_sha256 = _require_digest(document.get("model_sha256"), "model_sha256")
    pipeline_sha256 = _require_digest(
        document.get("pipeline_sha256"), "pipeline_sha256"
    )
    runtime_identity = _runtime_identity(document.get("runtime_identity"))

    input_path = _verified_member(root, document.get("input"))
    prediction_record = document.get("prediction")
    prediction_path = _verified_member(root, prediction_record)
    detections_record = document.get("detections")
    detections_path = _verified_member(root, detections_record)
    input_tensor = input_path.read_bytes()
    if len(input_tensor) != _INPUT_SIZE:
        raise OracleBundleError("input tensor size is invalid")
    try:
        predictions = np.load(prediction_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise OracleBundleError("prediction array cannot be loaded") from exc
    if predictions.dtype != np.float32 or predictions.shape != (13_566, 8):
        raise OracleBundleError("prediction matrix contract changed")
    if not np.all(np.isfinite(predictions)):
        raise OracleBundleError("prediction matrix contains non-finite values")
    expected_raw_digest = (
        prediction_record.get("raw_sha256")
        if isinstance(prediction_record, dict)
        else None
    )
    _require_digest(expected_raw_digest, "prediction raw_sha256")
    observed_raw_digest = hashlib.sha256(
        predictions.tobytes(order="C")
    ).hexdigest()
    if observed_raw_digest != expected_raw_digest:
        raise OracleBundleError("prediction raw SHA-256 mismatch")
    predictions.setflags(write=False)

    try:
        raw_detections = json.loads(detections_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleBundleError("detections cannot be loaded") from exc
    if not isinstance(raw_detections, list) or not all(
        isinstance(item, dict) for item in raw_detections
    ):
        raise OracleBundleError("detections member is not a list of objects")
    expected_detection_count = (
        detections_record.get("count") if isinstance(detections_record, dict) else None
    )
    if expected_detection_count != len(raw_detections):
        raise OracleBundleError("detection count mismatch")

    tensor_records = document.get("raw_tensors")
    if not isinstance(tensor_records, list) or len(tensor_records) != 22:
        raise OracleBundleError("oracle manifest must name exactly 22 raw tensors")
    tensors: dict[str, bytes] = {}
    for record in tensor_records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise OracleBundleError("raw tensor record is invalid")
        name = record["name"]
        if name in tensors:
            raise OracleBundleError(f"duplicate raw tensor name: {name}")
        member = _verified_member(root, record)
        tensors[name] = member.read_bytes()

    return OracleBundle(
        root=root,
        case_id=case_id,
        input_tensor=input_tensor,
        raw_tensors=MappingProxyType(tensors),
        predictions=predictions,
        detections=tuple(MappingProxyType(item) for item in raw_detections),
        model_sha256=model_sha256,
        pipeline_sha256=pipeline_sha256,
        runtime_identity=MappingProxyType(runtime_identity),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _runtime_identity(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise OracleBundleError("runtime_identity must be a non-empty object")
    result: dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name or not isinstance(item, str) or not item:
            raise OracleBundleError("runtime_identity fields must be non-empty strings")
        result[name] = item
    return dict(sorted(result.items()))


def _member_record(root: Path, path: Path, *, name: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if name is not None:
        result["name"] = name
    return result


def _verified_member(root: Path, value: object) -> Path:
    if not isinstance(value, dict):
        raise OracleBundleError("oracle member record is invalid")
    relative = value.get("path")
    size = value.get("size")
    digest = value.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise OracleBundleError("oracle member path is invalid")
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise OracleBundleError("oracle member escapes bundle root") from exc
    if candidate.is_symlink():
        raise OracleBundleError("oracle member must not be a symbolic link")
    if not candidate.is_file():
        raise OracleBundleError(f"missing oracle member: {relative}")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise OracleBundleError("oracle member size is invalid")
    if candidate.stat().st_size != size:
        raise OracleBundleError(f"oracle member size mismatch: {relative}")
    _require_digest(digest, "member SHA-256")
    if _sha256(candidate) != digest:
        raise OracleBundleError(f"oracle member SHA-256 mismatch: {relative}")
    return candidate


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise OracleBundleError(f"invalid {name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
