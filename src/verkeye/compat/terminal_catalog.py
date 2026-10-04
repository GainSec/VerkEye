"""Strict, deterministic catalogs for ADES terminal-tensor parity."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .ades import CavalryManifest, decode_float32_chw
from .oracle import OracleBundle, load_oracle_bundle


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_TERMINAL_SHAPES = {
    "output_0": [1, 4, 76, 136],
    "output_1": [1, 4, 38, 68],
    "output_2": [1, 4, 19, 34],
    "output_3": [1, 4, 76, 136],
    "output_4": [1, 4, 38, 68],
    "output_5": [1, 4, 19, 34],
}
_ORACLE_FIELDS = {
    "name",
    "container_image",
    "platform",
    "cavalry_version",
    "cavalry_hash",
}


class TerminalCatalogError(ValueError):
    """A parity or acquisition catalog violates its immutable contract."""


def validate_terminal_parity_catalog(
    document: object,
) -> tuple[Mapping[str, Any], ...]:
    """Validate a tracked terminal catalog and return its ordered cases."""

    root = _object(document, "terminal catalog")
    if root.get("schema") != "verkeye.ades-terminal-parity-catalog.v2":
        raise TerminalCatalogError("unsupported terminal catalog schema")
    _artifact(root.get("model"), "model")
    _artifact(root.get("pipeline"), "pipeline")
    _oracle(root.get("oracle"))
    cases = _cases(root)
    for case in cases:
        _digest(case.get("manifest_sha256"), "manifest_sha256")
        _digest(case.get("input_sha256"), "input_sha256")
        _digest(case.get("prediction_sha256"), "prediction_sha256")
        bundle = case.get("bundle")
        if not isinstance(bundle, str) or not bundle:
            raise TerminalCatalogError("case bundle must be a non-empty path")
        provenance = _object(case.get("provenance"), "case provenance")
        kind = provenance.get("kind")
        if not isinstance(kind, str) or not kind:
            raise TerminalCatalogError("case provenance kind is missing")
        terminals = _object(case.get("terminal_tensors"), "terminal_tensors")
        if set(terminals) != set(_TERMINAL_SHAPES):
            raise TerminalCatalogError("case must contain exactly six terminal tensors")
        for name, shape in _TERMINAL_SHAPES.items():
            record = _object(terminals[name], f"terminal tensor {name}")
            if record.get("dtype") != "float32" or record.get("shape") != shape:
                raise TerminalCatalogError(f"terminal tensor {name} contract changed")
            _digest(record.get("sha256"), f"terminal tensor {name} SHA-256")
        detections = case.get("production_detections")
        if not isinstance(detections, list) or not all(
            isinstance(item, dict) for item in detections
        ):
            raise TerminalCatalogError("production_detections must be a list of objects")
    return tuple(cases)


def validate_acquisition_document(
    document: object,
    *,
    terminal_catalog: object,
) -> None:
    """Require acquisition evidence to bind exactly one terminal catalog."""

    root = _object(document, "acquisition document")
    if root.get("schema") != "verkeye.ades-corpus-acquisition.v1":
        raise TerminalCatalogError("unsupported acquisition schema")
    terminal_root = _object(terminal_catalog, "terminal catalog")
    terminal_cases = validate_terminal_parity_catalog(terminal_root)
    _oracle(root.get("oracle"))
    if root.get("oracle") != terminal_root.get("oracle"):
        raise TerminalCatalogError("acquisition oracle identity differs")
    expected_ids = [case["case_id"] for case in terminal_cases]
    if root.get("case_ids") != expected_ids:
        raise TerminalCatalogError("acquisition case IDs differ")
    if root.get("case_count") != len(expected_ids):
        raise TerminalCatalogError("acquisition case count differs")
    categories = Counter(case["provenance"]["kind"] for case in terminal_cases)
    if root.get("category_counts") != dict(sorted(categories.items())):
        raise TerminalCatalogError("acquisition category counts differ")
    _digest(root.get("terminal_catalog_sha256"), "terminal catalog SHA-256")
    commands = root.get("commands")
    if not isinstance(commands, list) or not commands or not all(
        isinstance(item, str) and item for item in commands
    ):
        raise TerminalCatalogError("acquisition commands are missing")


def build_terminal_parity_catalog(
    source_catalog: object,
    *,
    manifest: CavalryManifest,
) -> dict[str, object]:
    """Build compact terminal hashes from strictly verified oracle bundles."""

    source = _object(source_catalog, "source oracle catalog")
    if source.get("schema") != "verkeye.ades-oracle-catalog.v2":
        raise TerminalCatalogError("unsupported source oracle catalog schema")
    model = _artifact(source.get("model"), "model")
    pipeline = _artifact(source.get("pipeline"), "pipeline")
    oracle = _oracle(source.get("oracle"))
    cases = _cases(source)
    terminal_cases: list[dict[str, object]] = []
    for record in cases:
        bundle_path = record.get("bundle")
        if not isinstance(bundle_path, str) or not bundle_path:
            raise TerminalCatalogError("case bundle must be a non-empty path")
        bundle = load_oracle_bundle(bundle_path)
        _bind_bundle(record, bundle, model=model, pipeline=pipeline, oracle=oracle)
        terminals = _terminal_tensors(bundle, manifest)
        terminal_cases.append(
            {
                "case_id": bundle.case_id,
                "bundle": bundle_path,
                "manifest_sha256": bundle.manifest_sha256,
                "provenance": dict(
                    _object(record.get("provenance"), "case provenance")
                ),
                "input_sha256": _sha256_bytes(bundle.input_tensor),
                "terminal_tensors": {
                    name: {
                        "dtype": "float32",
                        "shape": list(tensor.shape),
                        "sha256": _sha256_bytes(tensor.tobytes(order="C")),
                    }
                    for name, tensor in sorted(terminals.items())
                },
                "prediction_sha256": _sha256_bytes(
                    bundle.predictions.tobytes(order="C")
                ),
                "production_detections": [dict(item) for item in bundle.detections],
            }
        )
    result: dict[str, object] = {
        "schema": "verkeye.ades-terminal-parity-catalog.v2",
        "model": dict(model),
        "pipeline": dict(pipeline),
        "oracle": dict(oracle),
        "case_count": len(terminal_cases),
        "cases": terminal_cases,
    }
    validate_terminal_parity_catalog(result)
    return result


def canonical_json_bytes(document: object) -> bytes:
    """Serialize tracked evidence deterministically."""

    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _terminal_tensors(
    bundle: OracleBundle,
    manifest: CavalryManifest,
) -> dict[str, Any]:
    terminals: dict[str, Any] = {}
    for split in manifest.splits:
        for port in split.outputs:
            if not port.main_input_output:
                continue
            filename = f"split-{split.index:02d}-output-{port.index:02d}.bin"
            try:
                payload = bundle.raw_tensors[filename]
            except KeyError as error:
                raise TerminalCatalogError(
                    f"bundle {bundle.case_id} lacks terminal payload {filename}"
                ) from error
            if port.name in terminals:
                raise TerminalCatalogError(f"duplicate terminal tensor {port.name}")
            terminals[port.name] = decode_float32_chw(payload, port)
    if set(terminals) != set(_TERMINAL_SHAPES):
        raise TerminalCatalogError("Cavalry manifest does not expose six terminals")
    return terminals


def _bind_bundle(
    record: Mapping[str, Any],
    bundle: OracleBundle,
    *,
    model: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    oracle: Mapping[str, str],
) -> None:
    if record["case_id"] != bundle.case_id:
        raise TerminalCatalogError("bundle case_id differs from source catalog")
    expected = {
        "manifest_sha256": bundle.manifest_sha256,
        "input_sha256": _sha256_bytes(bundle.input_tensor),
        "prediction_sha256": _sha256_bytes(bundle.predictions.tobytes(order="C")),
    }
    for name, observed in expected.items():
        if record.get(name) != observed:
            raise TerminalCatalogError(f"bundle {bundle.case_id} {name} differs")
    if record.get("raw_tensor_count") != 22:
        raise TerminalCatalogError("source catalog raw tensor count differs")
    if record.get("detection_count") != len(bundle.detections):
        raise TerminalCatalogError("source catalog detection count differs")
    if bundle.model_sha256 != model["sha256"]:
        raise TerminalCatalogError("bundle model identity differs")
    if bundle.pipeline_sha256 != pipeline["sha256"]:
        raise TerminalCatalogError("bundle pipeline identity differs")
    if dict(bundle.runtime_identity) != dict(oracle):
        raise TerminalCatalogError("bundle oracle identity differs")


def _cases(root: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = root.get("cases")
    if not isinstance(raw, list) or root.get("case_count") != len(raw):
        raise TerminalCatalogError("case_count does not match cases")
    cases: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for value in raw:
        case = _object(value, "case")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
            raise TerminalCatalogError("invalid case_id")
        if case_id in seen:
            raise TerminalCatalogError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        cases.append(case)
    return cases


def _artifact(value: object, name: str) -> Mapping[str, str]:
    artifact = _object(value, name)
    if set(artifact) != {"path", "sha256"}:
        raise TerminalCatalogError(f"{name} identity fields differ")
    path = artifact.get("path")
    if not isinstance(path, str) or not path:
        raise TerminalCatalogError(f"{name} path is missing")
    _digest(artifact.get("sha256"), f"{name} SHA-256")
    return artifact  # type: ignore[return-value]


def _oracle(value: object) -> Mapping[str, str]:
    oracle = _object(value, "oracle identity")
    if set(oracle) != _ORACLE_FIELDS or not all(
        isinstance(item, str) and item for item in oracle.values()
    ):
        raise TerminalCatalogError("oracle identity fields differ")
    return oracle  # type: ignore[return-value]


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TerminalCatalogError(f"{name} must be an object")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TerminalCatalogError(f"invalid {name}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
