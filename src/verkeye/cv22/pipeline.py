"""Pinned production-pipeline evidence recovered from the CB62 ``cvproc``.

This module deliberately separates two claims:

* the raw detector-head and core post-processing behavior that is proved by
  the exact recovered executable; and
* the complete camera-free pipeline gate, including the exact recovered
  model-boundary preprocessing contract.

Constructor defaults remain identified as defaults.  Operational values are
reported only after applying the exact recovered firmware merge semantics to
the exact recovered production configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any

from .vendor_runtime import VendorRuntimeError, analyze_elf


class PipelineEvidenceError(ValueError):
    """The exact recovered artifacts or a pinned proof no longer match."""


_ARTIFACTS = {
    "cvproc": {
        "size": 6_773_760,
        "sha256": "47aad1ba15b23851ad606ba7ba2e92ceac9289176b0eacf0d124020f5606ba7b",
    },
    "libvproc": {
        "size": 202_616,
        "sha256": "f7d0fbe415db5fccef15bbc714ed3c5be563d098bc83f4e3f23d381f4e24e310",
    },
    "model": {
        "size": 5_561_124,
        "sha256": "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf",
    },
    "production_config": {
        "size": 14_028,
        "sha256": "022d8371a822b702cd7c0cbc81b7280a84b2f724fb52ab8352fd92a38f346908",
    },
    "vconfig": {
        "size": 225_428,
        "sha256": "32609cc9ea84f9f2887c951a33b432177b6592e594ba458eae088c3b52900255",
    },
    "bruce_4k": {
        "size": 5_420,
        "sha256": "9aa2012ffa2ed1ae2683a1b3c0c825187b470446e2ba584e07f91e8a70e5ecbd",
    },
    "bruce_4k_telephoto": {
        "size": 5_819,
        "sha256": "d91f20143d3635f1fa471473ff36764ab0d84b10bfb2778027363ebd899d2c46",
    },
}

_FUNCTIONS = {
    "grid_constructor": {
        "virtual_address": 0x4FD450,
        "size": 5_188,
        "sha256": "c1cd73ecf7be12239e1b70df341ca95c440d2f35492b8903b72b789423c25151",
    },
    "secondary_dedup": {
        "virtual_address": 0x4FB0E0,
        "size": 1_792,
        "sha256": "ed6a5c6151cb6babdfae3bc15a8026add303ab4197deff58dabb4e1fb395b131",
    },
    "postprocess": {
        "virtual_address": 0x5017E0,
        "size": 3_780,
        "sha256": "baed2630aeb78127b5c3cd436b806dd0faca7f67442cae482c56fc610b1b4b8e",
    },
    "iou_distance": {
        "virtual_address": 0x637F50,
        "size": 888,
        "sha256": "448630416994b78c7fee1307d13cbb7cc13052d174862ebb7fe13a5d1fbf4fec",
    },
    "minimum": {
        "virtual_address": 0x4F4340,
        "size": 344,
        "sha256": "3ebc1531a65877cd1e07256898b228dfb9e4e0ed723b42718768175f346e1e35",
    },
    "config_update": {
        "virtual_address": 0x4DAE30,
        "size": 6_236,
        "sha256": "dd5e7ae1f51244aa555e2a391d0b2a14966a2fc8b7ad76bbccf04bd8d7f06544",
    },
    "detection_consumer": {
        "virtual_address": 0x506330,
        "size": 2_872,
        "sha256": "9fedea6cee4352c0c04bfa440bee5d2b9d6a29ca010ab250415efb0f72486b54",
    },
}

_RODATA = {
    "constructor_defaults_a": {
        "virtual_address": 0x940A90,
        "size": 32,
        "hex": (
            "6f12833a0ad7233bcdcccc3d00002041"
            "cdcccc3db81e053e0000803e0000003f"
        ),
    },
    "constructor_defaults_b": {
        "virtual_address": 0x9409E0,
        "size": 8,
        "hex": "6666263f0000003f",
    },
    "strides": {
        "virtual_address": 0x940FB8,
        "size": 8,
        "hex": "0800000010000000",
    },
    "class_person": {
        "virtual_address": 0x975AE8,
        "size": 7,
        "hex": "706572736f6e00",
    },
    "class_vehicle": {
        "virtual_address": 0x975B18,
        "size": 8,
        "hex": "76656869636c6500",
    },
    "class_animal": {
        "virtual_address": 0x9357D0,
        "size": 7,
        "hex": "616e696d616c00",
    },
}

_PREPROCESS_FUNCTIONS = {
    "cvproc_functions": {
        "ordinary_converter": {
            "virtual_address": 0x4F1ED0,
            "size": 0x268,
            "sha256": "893d061acb99c5dfdd464190e0141287b73af0ad3566f19a3afa871baacc924e",
        },
        "converter_setup": {
            "virtual_address": 0x4F24D0,
            "size": 0x2BC,
            "sha256": "49ddd01f0b09aa1e8f4c0cb1fc2840cc6feae6068a152740573999f424ffc7ed",
        },
        "pipeline_dispatcher": {
            "virtual_address": 0x4F3B80,
            "size": 0x15C,
            "sha256": "0293480f7a4fee9cc9357c2496717f1f8d507fce38ed31169ac4ed34ee7bd148",
        },
    },
    "libvproc_functions": {
        "image_deformation": {
            "virtual_address": 0x8CC8,
            "size": 0x434,
            "sha256": "68772a0f8144f9e8e93752a78d94dfb48e640c09baa3ed835a0649c3b0e0496f",
        },
        "resize": {
            "virtual_address": 0x152C4,
            "size": 0xE4,
            "sha256": "e0d5f3fbfc342934f99cbdfdac0ef746157f1d1a37471729890e789364828ab7",
        },
        "yuv2rgb_420": {
            "virtual_address": 0x20E34,
            "size": 0xE40,
            "sha256": "2d098194d1e854512a4d2a63ad76c98af84c6ff46dbc7b0d78cffb6d16a5d7a5",
        },
    },
}

_PREPROCESS_RODATA = {
    "target_descriptor_shuffle": {
        "virtual_address": 0x940BF0,
        "size": 16,
        "hex": "00010203040506070c0d0e0f00010203",
    }
}

_INPUT_SHAPE = (1, 3, 608, 1088)
_STRIDES = (8, 16, 32)
_RAW_COLUMNS = 8
_RAW_ROW_BYTES = _RAW_COLUMNS * 4
_RAW_BUFFER_BYTES = 0x69FC0
_UNRESOLVED: list[str] = []

_OPERATIONAL_DEFAULTS = {
    "dedup-iou-threshold": 0.8,
    "dedup-vehicle-enabled": True,
    "max-from-last-dedup-time-sec": 60,
    "max-from-start-dedup-time-sec": 300,
    "motion-animal-threshold": 0.4,
    "motion-person-threshold": 0.5,
    "motion-vehicle-threshold": 0.5,
    "person-threshold": 0.5,
    "vehicle-threshold": 0.5,
}

_CVPROC_BINDINGS = [
    {
        "key": "person-threshold",
        "getter_virtual_address": 0x728A80,
        "destination_offset": 0x2E14,
        "member": "mPersonThreshold",
        "type": "float",
    },
    {
        "key": "vehicle-threshold",
        "getter_virtual_address": 0x728C90,
        "destination_offset": 0x2E18,
        "member": "mVehicleThreshold",
        "type": "float",
    },
    {
        "key": "motion-person-threshold",
        "getter_virtual_address": 0x727FE0,
        "destination_offset": 0x2E08,
        "member": "mMotionPersonThreshold",
        "type": "float",
    },
    {
        "key": "motion-vehicle-threshold",
        "getter_virtual_address": 0x728020,
        "destination_offset": 0x2E0C,
        "member": "mMotionVehicleThreshold",
        "type": "float",
    },
    {
        "key": "motion-animal-threshold",
        "getter_virtual_address": 0x727FC0,
        "destination_offset": 0x2E10,
        "member": "mMotionAnimalThreshold",
        "type": "float",
    },
    {
        "key": "dedup-vehicle-enabled",
        "getter_virtual_address": 0x7271C0,
        "destination_offset": 0x2E3C,
        "member": "mDedupVehicleEnabled",
        "type": "bool",
    },
    {
        "key": "dedup-iou-threshold",
        "getter_virtual_address": 0x7271A0,
        "destination_offset": 0x2E40,
        "member": "mDedupIouThreshold",
        "type": "float",
    },
    {
        "key": "max-from-start-dedup-time-sec",
        "getter_virtual_address": 0x727F20,
        "destination_offset": 0x2E44,
        "member": "mMaxFromStartDedupTimeSec",
        "type": "int",
    },
    {
        "key": "max-from-last-dedup-time-sec",
        "getter_virtual_address": 0x727F00,
        "destination_offset": 0x2E48,
        "member": "mMaxFromLastDedupTimeSec",
        "type": "int",
    },
]


def _read_pinned(path: str | Path, role: str) -> tuple[Path, bytes, dict[str, Any]]:
    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise PipelineEvidenceError(f"cannot read {role} artifact {source}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    expected = _ARTIFACTS[role]
    if len(raw) != expected["size"] or digest != expected["sha256"]:
        raise PipelineEvidenceError(
            f"{role} artifact mismatch: expected size {expected['size']} sha256 "
            f"{expected['sha256']}, observed size {len(raw)} sha256 {digest}"
        )
    return source, raw, {"path": str(source), "size": len(raw), "sha256": digest}


def _vaddr_bytes(
    raw: bytes,
    elf: dict[str, Any],
    virtual_address: int,
    size: int,
) -> tuple[int, bytes]:
    for section in elf["sections"]:
        start = section["virtual_address"]
        end = start + section["size"]
        if start <= virtual_address and virtual_address + size <= end:
            file_offset = section["file_offset"] + virtual_address - start
            return file_offset, raw[file_offset : file_offset + size]
    raise PipelineEvidenceError(
        f"virtual range 0x{virtual_address:x}+0x{size:x} is outside ELF sections"
    )


def _verify_binary_proofs(cvproc_path: Path, raw: bytes) -> dict[str, Any]:
    try:
        elf = analyze_elf(cvproc_path)
    except VendorRuntimeError as exc:
        raise PipelineEvidenceError(f"cannot analyze pinned cvproc ELF: {exc}") from exc

    functions: dict[str, dict[str, Any]] = {}
    for name, expected in _FUNCTIONS.items():
        _, body = _vaddr_bytes(
            raw,
            elf,
            expected["virtual_address"],
            expected["size"],
        )
        digest = hashlib.sha256(body).hexdigest()
        if digest != expected["sha256"]:
            raise PipelineEvidenceError(
                f"{name} binary proof mismatch at "
                f"0x{expected['virtual_address']:x}: expected {expected['sha256']}, "
                f"observed {digest}"
            )
        functions[name] = dict(expected)

    rodata: dict[str, dict[str, Any]] = {}
    for name, expected in _RODATA.items():
        _, observed = _vaddr_bytes(
            raw,
            elf,
            expected["virtual_address"],
            expected["size"],
        )
        if observed.hex() != expected["hex"]:
            raise PipelineEvidenceError(
                f"{name} rodata proof mismatch at "
                f"0x{expected['virtual_address']:x}: expected {expected['hex']}, "
                f"observed {observed.hex()}"
            )
        rodata[name] = dict(expected)

    rodata["strides"]["decoded_int32_le"] = list(
        struct.unpack("<2i", bytes.fromhex(rodata["strides"]["hex"]))
    )
    rodata["strides"]["third_value"] = {
        "value": 32,
        "evidence": "immediate stored by the pinned grid constructor",
    }
    return {"functions": functions, "rodata": rodata}


def _verify_function_ranges(
    artifact_name: str,
    path: Path,
    raw: bytes,
    expected_functions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    try:
        elf = analyze_elf(path)
    except VendorRuntimeError as exc:
        raise PipelineEvidenceError(
            f"cannot analyze pinned {artifact_name} ELF: {exc}"
        ) from exc

    observed_functions: dict[str, dict[str, Any]] = {}
    for name, expected in expected_functions.items():
        _, body = _vaddr_bytes(
            raw,
            elf,
            expected["virtual_address"],
            expected["size"],
        )
        digest = hashlib.sha256(body).hexdigest()
        if digest != expected["sha256"]:
            raise PipelineEvidenceError(
                f"{artifact_name} {name} binary proof mismatch at "
                f"0x{expected['virtual_address']:x}: expected {expected['sha256']}, "
                f"observed {digest}"
            )
        observed_functions[name] = dict(expected)
    return observed_functions


def _verify_preprocess_binary_proofs(
    cvproc_path: Path,
    cvproc_raw: bytes,
    libvproc_path: Path,
    libvproc_raw: bytes,
) -> dict[str, Any]:
    cvproc_functions = _verify_function_ranges(
        "cvproc",
        cvproc_path,
        cvproc_raw,
        _PREPROCESS_FUNCTIONS["cvproc_functions"],
    )
    libvproc_functions = _verify_function_ranges(
        "libvproc",
        libvproc_path,
        libvproc_raw,
        _PREPROCESS_FUNCTIONS["libvproc_functions"],
    )
    try:
        cvproc_elf = analyze_elf(cvproc_path)
    except VendorRuntimeError as exc:  # pragma: no cover - verified above
        raise PipelineEvidenceError(f"cannot analyze pinned cvproc ELF: {exc}") from exc

    rodata: dict[str, dict[str, Any]] = {}
    for name, expected in _PREPROCESS_RODATA.items():
        _, observed = _vaddr_bytes(
            cvproc_raw,
            cvproc_elf,
            expected["virtual_address"],
            expected["size"],
        )
        if observed.hex() != expected["hex"]:
            raise PipelineEvidenceError(
                f"{name} preprocessing rodata mismatch at "
                f"0x{expected['virtual_address']:x}: expected {expected['hex']}, "
                f"observed {observed.hex()}"
            )
        rodata[name] = dict(expected)
    return {
        "cvproc_functions": cvproc_functions,
        "libvproc_functions": libvproc_functions,
        "cvproc_rodata": rodata,
    }


def _preprocessing_report() -> dict[str, Any]:
    return {
        "status": "passed",
        "source": {
            "pixel_format": "NV12",
            "width": 1088,
            "height": 608,
            "pitch": 1088,
            "cvproc_color_enum": 2,
            "libvproc_internal_format": 10,
        },
        "geometry": {
            "operation": "identity",
            "source_size": [608, 1088],
            "target_size": [608, 1088],
            "local_policy": "reject-nonmatching-geometry",
        },
        "color_conversion": {
            "matrix": "full-range BT.601",
            "formula": {
                "r": "Y + 1.4019999504089355 * (V - 128)",
                "g": (
                    "Y - 0.34413599967956543 * (U - 128) "
                    "- 0.714136004447937 * (V - 128)"
                ),
                "b": "Y + 1.7719999551773071 * (U - 128)",
            },
            "output_channel_order": ["B", "G", "R"],
        },
        "tensor": {
            "shape": [1, 3, 608, 1088],
            "storage_dtype": "uint8",
            "layout": "NCHW-planar-contiguous",
            "channel_order": ["B", "G", "R"],
            "normalization": "none",
            "byte_count": 1_984_512,
        },
        "local_input_contract": {
            "accepted": "1088x608 decoded uint8 BGR image",
            "operation": "planarize without resize, reorder, scaling, or normalization",
            "camera_source_parity": "pending physical-camera fixture comparison",
        },
    }


def _load_root_input(tensor_map_path: str | Path, model_digest: str) -> dict[str, Any]:
    source = Path(tensor_map_path).resolve()
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineEvidenceError(f"cannot read tensor map {source}: {exc}") from exc
    artifact = report.get("artifact", {})
    if artifact.get("sha256") != model_digest:
        raise PipelineEvidenceError(
            "tensor-map model digest does not match the pinned model artifact"
        )
    external = report.get("external_inputs")
    splits = report.get("splits")
    if external != [{"name": "images", "offset": 1572, "ordinal": 0, "split": 0}]:
        raise PipelineEvidenceError("tensor map does not contain the exact root images input")
    if not isinstance(splits, list) or not splits:
        raise PipelineEvidenceError("tensor map lacks split records")
    try:
        root = splits[0]["inputs"][0]
        shape = tuple(item["value"] for item in root["dimensions"])
        storage_dtype = root["data_format"]["storage_dtype"]
        name = root["name"]["value"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PipelineEvidenceError("tensor map root descriptor is malformed") from exc
    if shape != _INPUT_SHAPE or storage_dtype != "uint8" or name != "images":
        raise PipelineEvidenceError(
            f"unexpected root descriptor: name={name!r} shape={shape!r} "
            f"storage_dtype={storage_dtype!r}"
        )
    return {
        "shape": list(shape),
        "height": shape[2],
        "width": shape[3],
        "storage_dtype": storage_dtype,
        "evidence": "exact recovered tensor descriptor named images",
    }


def _grid_report(height: int, width: int) -> dict[str, Any]:
    levels: list[dict[str, int]] = []
    start = 0
    for stride in _STRIDES:
        grid_width = width // stride
        grid_height = height // stride
        count = grid_width * grid_height
        levels.append(
            {
                "stride": stride,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "start_row": start,
                "row_count": count,
            }
        )
        start += count
    if start * _RAW_ROW_BYTES != _RAW_BUFFER_BYTES:
        raise PipelineEvidenceError(
            "recovered grid dimensions do not account for the exact output buffer"
        )
    return {
        "order": "stride-level, then y-major, then x-minor",
        "levels": levels,
        "total_rows": start,
    }


def _constructor_defaults(proofs: dict[str, Any]) -> dict[str, Any]:
    first = bytes.fromhex(proofs["rodata"]["constructor_defaults_a"]["hex"])
    second = bytes.fromhex(proofs["rodata"]["constructor_defaults_b"]["hex"])
    values_a = struct.unpack("<8f", first)
    values_b = struct.unpack("<2f", second)
    return {
        "disposition": "exact-constructor-defaults",
        "operational_override_status": "resolved",
        "values": {
            "maximum_detections": 30,
            "minimum_normalized_area": values_a[0],
            "border_margin": values_a[1],
            "minimum_aspect_ratio": values_a[2],
            "maximum_aspect_ratio": values_a[3],
            "class_0_1_threshold_primary": values_a[4],
            "class_0_1_threshold_alternate": values_a[5],
            "class_2_threshold": values_a[6],
            "routing_threshold_primary": values_a[7],
            "routing_threshold_alternate": values_b[0],
            "nms_distance_threshold": values_b[1],
        },
        "claim_boundary": (
            "the pinned executable initializes these values; separately reported "
            "operational values apply the exact same-firmware default/product/"
            "persisted configuration merge and cvproc bindings"
        ),
    }


def _class_mapping() -> dict[str, Any]:
    """Report the class-ID dataflow proved by the pinned executable.

    The postprocessor's outer loop emits its current class ID through the
    detection factory.  The downstream consumer loads that same detection
    field at offset 0x18 and branches to literal person, vehicle, and animal
    label strings for IDs zero, one, and two respectively.
    """

    return {
        "status": "passed",
        "labels": [
            {
                "class_id": 0,
                "label": "person",
                "string_virtual_address": 0x975AE8,
                "consumer_branch_virtual_address": 0x506460,
            },
            {
                "class_id": 1,
                "label": "vehicle",
                "string_virtual_address": 0x975B18,
                "consumer_branch_virtual_address": 0x506468,
            },
            {
                "class_id": 2,
                "label": "animal",
                "string_virtual_address": 0x9357D0,
                "consumer_branch_virtual_address": 0x506470,
            },
        ],
        "producer": {
            "class_loop_virtual_address": 0x501B28,
            "class_id_stack_store_virtual_address": 0x501B30,
            "class_id_factory_argument_load_virtual_address": 0x501DB8,
            "factory_call_virtual_address": 0x501DD8,
            "evidence": (
                "the pinned postprocessor iterates class IDs 0,1,2 and passes "
                "that ID as argument w3 to the detection factory"
            ),
        },
        "consumer": {
            "function_virtual_address": 0x506330,
            "class_field_offset": 0x18,
            "class_field_load_virtual_address": 0x50645C,
            "evidence": (
                "the pinned downstream detection consumer loads the emitted "
                "class field and selects the exact label strings"
            ),
        },
    }


def _operational_configuration(
    production_config_raw: bytes,
    vconfig_raw: bytes,
    bruce_4k_raw: bytes,
    bruce_4k_telephoto_raw: bytes,
) -> dict[str, Any]:
    """Apply the recovered configuration precedence to the relevant CV fields."""

    try:
        configuration = json.loads(production_config_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineEvidenceError(f"production config is not valid JSON: {exc}") from exc
    if not isinstance(configuration, dict):
        raise PipelineEvidenceError("production config root is not an object")

    try:
        system_version = configuration["system-config"]["sys-version"]
        persisted = configuration["cvproc-hyperzoom"]
    except (KeyError, TypeError) as exc:
        raise PipelineEvidenceError(
            "production config lacks system version or cvproc-hyperzoom"
        ) from exc
    if system_version != "v2026.07.30.926397-bruce-4k":
        raise PipelineEvidenceError(
            f"unexpected production system version: {system_version!r}"
        )
    if not isinstance(persisted, dict):
        raise PipelineEvidenceError("cvproc-hyperzoom config is not an object")

    required_vconfig_tokens = [
        *[key.encode("ascii") for key in _OPERATIONAL_DEFAULTS],
        b"get_default_config",
        b"override_config",
        b"read_config",
    ]
    missing_tokens = [
        token.decode("ascii")
        for token in required_vconfig_tokens
        if token not in vconfig_raw
    ]
    if missing_tokens:
        raise PipelineEvidenceError(
            "vconfig bytecode lacks pinned configuration symbols: "
            + ", ".join(missing_tokens)
        )

    relevant_tokens = [key.encode("ascii") for key in _OPERATIONAL_DEFAULTS]
    product_payloads = {
        "bruce-4k": bruce_4k_raw,
        "bruce-4k-telephoto": bruce_4k_telephoto_raw,
    }
    product_overrides = {
        name: [
            token.decode("ascii") for token in relevant_tokens if token in payload
        ]
        for name, payload in product_payloads.items()
    }
    if any(product_overrides.values()):
        raise PipelineEvidenceError(
            "pinned product configuration unexpectedly overrides relevant keys"
        )

    effective: dict[str, dict[str, Any]] = {}
    for key, default in _OPERATIONAL_DEFAULTS.items():
        if key in persisted:
            effective[key] = {
                "value": persisted[key],
                "source": "persisted_override",
            }
        else:
            effective[key] = {"value": default, "source": "firmware_default"}

    return {
        "system_version": system_version,
        "product_profile": "bruce-4k",
        "merge_order": [
            "firmware_defaults",
            "product_defaults",
            "persisted_config",
        ],
        "effective": effective,
        "product_overrides": {
            **product_overrides,
            "disposition": "no relevant keys in either pinned product profile",
        },
        "merge_proof": {
            "ServiceConfig.default": "validator entry element 0",
            "Config.get_default_config": "all ServiceConfig subclasses",
            "Config.read_config": (
                "firmware defaults, then product config, then /mnt/config/config"
            ),
            "Config.override_config": "recursive source-over-destination overwrite",
        },
        "cvproc_bindings": [dict(binding) for binding in _CVPROC_BINDINGS],
    }


def analyze_cvproc_pipeline(
    cvproc: str | Path,
    libvproc: str | Path,
    model: str | Path,
    tensor_map: str | Path,
    production_config: str | Path,
    vconfig: str | Path,
    bruce_4k: str | Path,
    bruce_4k_telephoto: str | Path,
) -> dict[str, Any]:
    """Verify and report the exact recovered raw-head/postprocessor contract."""

    cvproc_path, cvproc_raw, cvproc_artifact = _read_pinned(cvproc, "cvproc")
    libvproc_path, libvproc_raw, libvproc_artifact = _read_pinned(
        libvproc, "libvproc"
    )
    _, _, model_artifact = _read_pinned(model, "model")
    _, production_config_raw, production_config_artifact = _read_pinned(
        production_config, "production_config"
    )
    _, vconfig_raw, vconfig_artifact = _read_pinned(vconfig, "vconfig")
    _, bruce_4k_raw, bruce_4k_artifact = _read_pinned(bruce_4k, "bruce_4k")
    _, bruce_4k_telephoto_raw, bruce_4k_telephoto_artifact = _read_pinned(
        bruce_4k_telephoto, "bruce_4k_telephoto"
    )
    input_record = _load_root_input(tensor_map, model_artifact["sha256"])
    proofs = _verify_binary_proofs(cvproc_path, cvproc_raw)
    preprocess_proofs = _verify_preprocess_binary_proofs(
        cvproc_path,
        cvproc_raw,
        libvproc_path,
        libvproc_raw,
    )
    grid = _grid_report(input_record["height"], input_record["width"])
    operational_configuration = _operational_configuration(
        production_config_raw,
        vconfig_raw,
        bruce_4k_raw,
        bruce_4k_telephoto_raw,
    )

    return {
        "schema": "verkeye.cv22.cvproc-pipeline.v1",
        "artifacts": {
            "cvproc": cvproc_artifact,
            "libvproc": libvproc_artifact,
            "model": model_artifact,
            "production_config": production_config_artifact,
            "vconfig": vconfig_artifact,
            "bruce_4k": bruce_4k_artifact,
            "bruce_4k_telephoto": bruce_4k_telephoto_artifact,
        },
        "input": input_record,
        "preprocessing": _preprocessing_report(),
        "raw_output": {
            "rows": grid["total_rows"],
            "columns": _RAW_COLUMNS,
            "element_type": "float32",
            "row_bytes": _RAW_ROW_BYTES,
            "buffer_bytes": _RAW_BUFFER_BYTES,
            "layout": [
                "raw_x",
                "raw_y",
                "raw_log_w",
                "raw_log_h",
                "objectness",
                "class_0",
                "class_1",
                "class_2",
            ],
        },
        "grid": grid,
        "postprocess_core": {
            "gate": {"status": "passed", "reason_codes": []},
            "confidence": {
                "class_selection": "maximum of exactly three class scores",
                "tie_break": "first class index",
                "formula": "objectness * selected_class_score",
                "threshold_comparison": "strictly greater than class threshold",
                "candidate_order": (
                    "class IDs 0,1,2; within class descending confidence, "
                    "then ascending source row"
                ),
            },
            "box_decode": {
                "center_x": "stride * (grid_x + raw_x)",
                "center_y": "stride * (grid_y + raw_y)",
                "width": "stride * exp(raw_log_w)",
                "height": "stride * exp(raw_log_h)",
                "clip": "input bounds before normalization",
                "normalize": "x / input_width and y / input_height",
            },
            "filters": {
                "minimum_area": "normalized width * normalized height",
                "border_or_aspect": (
                    "accept boxes wholly inside the border margin OR inside "
                    "the inclusive height/width aspect-ratio bounds"
                ),
                "non_finite": "reject with explicit NaN error path",
            },
            "nms": {
                "distance": "1 - IoU",
                "accept": "nms_distance_threshold <= minimum_distance",
                "equality": "accepted",
            },
        },
        "constructor_defaults": _constructor_defaults(proofs),
        "operational_configuration": operational_configuration,
        "class_mapping": _class_mapping(),
        "binary_proofs": proofs,
        "preprocess_binary_proofs": preprocess_proofs,
        "exact_pipeline_gate": {
            "status": "passed" if not _UNRESOLVED else "blocked",
            "reason_codes": list(_UNRESOLVED),
        },
        "claim_boundary": (
            "the pinned binaries prove exact production identity-geometry "
            "NV12-to-planar-BGR uint8 preprocessing, raw-head geometry, and the core "
            "postprocessor; the pinned same-firmware configuration resolves "
            "operational thresholds and secondary deduplication parameters; "
            "the downstream consumer proves the class-ID label mapping; physical-camera "
            "fixtures remain necessary only for independent parity comparison"
        ),
    }


def require_exact_pipeline(report: dict[str, Any]) -> None:
    """Reject a report unless every computation-affecting pipeline gate passes."""

    gate = report.get("exact_pipeline_gate")
    if not isinstance(gate, dict) or gate.get("status") != "passed":
        reasons = gate.get("reason_codes", []) if isinstance(gate, dict) else []
        detail = ", ".join(str(reason) for reason in reasons) or "missing gate"
        raise PipelineEvidenceError(f"exact pipeline is not ready: {detail}")
