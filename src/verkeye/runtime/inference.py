"""End-to-end exact recovered CB62 image inference."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from ..compat.ades_runtime import AdesInference, AdesRuntimeSpec
from ..ir import Disposition
from .preprocess import EvidenceField, preprocess_cb62_exact_bgr
from .sources import MediaFrame
from .yolov6 import (
    CB62CoreDetection,
    CB62RawHeadProfile,
    decode_cb62_raw_head_core,
)


class ExactImageInferenceError(RuntimeError):
    """Raised when an exact image-inference gate cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class ExactFrameResult:
    """One exact model execution with typed and serialized detections."""

    document: dict[str, object]
    detections: tuple[CB62CoreDetection, ...]


@dataclass(frozen=True, slots=True)
class ExactLiveFrameResult:
    """Bounded live result that omits unbounded forensic serialization."""

    run_id: str
    detections: tuple[CB62CoreDetection, ...]
    backend: dict[str, object]
    fidelity: dict[str, object]
    timings_ns: dict[str, int]


class ExactInferenceBackend(Protocol):
    def infer(self, input_tensor: bytes, *, run_id: str) -> AdesInference: ...


_PIPELINE_SHA256 = (
    "ffd295e8382bf29877f4c54be1d9a37948531fc2d16c22c6528b30dbe598b405"
)
_RAW_LAYOUT = (
    "raw_x",
    "raw_y",
    "raw_log_w",
    "raw_log_h",
    "objectness",
    "class_0",
    "class_1",
    "class_2",
)
_LABELS = {0: "person", 1: "vehicle", 2: "animal"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(value: object, evidence: str) -> EvidenceField[object]:
    return EvidenceField(value, Disposition.EXACT, evidence)


def load_cb62_core_profile(
    path: str | Path,
    *,
    expected_sha256: str = _PIPELINE_SHA256,
    operating_profile: str = "production",
) -> CB62RawHeadProfile:
    """Load the binary-proved core postprocessor from pinned evidence."""

    source = Path(path)
    try:
        digest = _sha256(source)
        document = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ExactImageInferenceError(
            f"cannot load pipeline evidence {source}"
        ) from error
    if digest != expected_sha256:
        raise ExactImageInferenceError(
            "pipeline evidence SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {digest}"
        )
    if document.get("schema") != "verkeye.cv22.cvproc-pipeline.v1":
        raise ExactImageInferenceError("unsupported pipeline evidence schema")
    if document.get("exact_pipeline_gate") != {
        "status": "passed",
        "reason_codes": [],
    }:
        raise ExactImageInferenceError("exact pipeline evidence gate is not passed")
    try:
        input_shape = document["input"]["shape"]
        strides = tuple(level["stride"] for level in document["grid"]["levels"])
        raw = document["raw_output"]
        defaults = document["constructor_defaults"]["values"]
        class_labels = {
            item["class_id"]: item["label"]
            for item in document["class_mapping"]["labels"]
        }
    except (KeyError, TypeError) as error:
        raise ExactImageInferenceError("pipeline evidence is incomplete") from error
    if input_shape != [1, 3, 608, 1088] or strides != (8, 16, 32):
        raise ExactImageInferenceError("pipeline input/grid contract changed")
    if (
        raw.get("rows") != 13_566
        or raw.get("columns") != 8
        or raw.get("element_type") != "float32"
        or tuple(raw.get("layout", ())) != _RAW_LAYOUT
    ):
        raise ExactImageInferenceError("pipeline raw-output contract changed")
    if class_labels != _LABELS:
        raise ExactImageInferenceError("pipeline class mapping changed")

    proof = (
        f"{source.as_posix()} sha256={digest}; pinned cvproc postprocess "
        "function 0x5017e0"
    )
    primary = np.float32(defaults["class_0_1_threshold_primary"])
    profile = CB62RawHeadProfile(
        input_size=_exact((608, 1088), proof),
        strides=_exact((8, 16, 32), proof),
        raw_layout=_exact(_RAW_LAYOUT, proof),
        class_thresholds=_exact(
            (
                primary,
                primary,
                np.float32(defaults["class_2_threshold"]),
            ),
            proof + "; exact primary constructor path",
        ),
        nms_distance_threshold=_exact(
            np.float32(defaults["nms_distance_threshold"]), proof
        ),
        maximum_detections=_exact(int(defaults["maximum_detections"]), proof),
        minimum_normalized_area=_exact(
            np.float32(defaults["minimum_normalized_area"]), proof
        ),
        border_margin=_exact(np.float32(defaults["border_margin"]), proof),
        aspect_ratio_bounds=_exact(
            (
                np.float32(defaults["minimum_aspect_ratio"]),
                np.float32(defaults["maximum_aspect_ratio"]),
            ),
            proof,
        ),
    )
    if operating_profile == "production":
        return profile  # type: ignore[return-value]
    if operating_profile == "small-object":
        return replace(
            profile,
            minimum_normalized_area=EvidenceField(
                np.float32(0.00025),
                Disposition.INFERRED,
                "local small-object operating profile; lowers only the host-side "
                "minimum normalized area gate from 0.001 to 0.00025 while the "
                "recovered model execution and all other postprocessor fields "
                "remain unchanged",
            ),
        )
    raise ValueError(f"unsupported CB62 operating profile: {operating_profile}")


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _execute_exact_decoded(
    *,
    model_path: Path,
    decoded: np.ndarray,
    image_record: dict[str, object],
    backend: ExactInferenceBackend,
    runtime_spec: AdesRuntimeSpec,
    profile: CB62RawHeadProfile,
    operating_profile: str,
    run_id: str,
    source_decode_ns: int | None = None,
    model_record: dict[str, object] | None = None,
) -> ExactFrameResult:
    preprocess_started = time.monotonic_ns()
    try:
        prepared = preprocess_cb62_exact_bgr(decoded)
    except ValueError as error:
        raise ExactImageInferenceError(str(error)) from error
    preprocess_ns = time.monotonic_ns() - preprocess_started
    execution = backend.infer(prepared.tensor.tobytes(), run_id=run_id)
    postprocess_started = time.monotonic_ns()
    detections = decode_cb62_raw_head_core(execution.predictions, profile)
    postprocess_ns = time.monotonic_ns() - postprocess_started

    tensor_records: list[dict[str, object]] = []
    for split in execution.execution.splits:
        for tensor in split.tensors:
            filename = Path(tensor.path).name
            if filename not in execution.tensor_sha256:
                raise ExactImageInferenceError(
                    f"backend omitted tensor digest for {filename}"
                )
            tensor_records.append(
                {
                    "split_index": tensor.split_index,
                    "output_index": tensor.output_index,
                    "name": tensor.name,
                    "file": filename,
                    "size": tensor.size,
                    "changed_bytes": tensor.changed_bytes,
                    "main_input_output": tensor.main_input_output,
                    "sha256": execution.tensor_sha256[filename],
                }
            )
    prediction_bytes = execution.predictions.tobytes(order="C")
    serialized_detections = []
    for detection in detections:
        x1, y1, x2, y2 = detection.normalized_box
        serialized_detections.append(
            {
                "class_id": detection.class_id,
                "label": _LABELS[detection.class_id],
                "confidence": detection.confidence,
                "normalized_box": [x1, y1, x2, y2],
                "pixel_box": [
                    x1 * decoded.shape[1],
                    y1 * decoded.shape[0],
                    x2 * decoded.shape[1],
                    y2 * decoded.shape[0],
                ],
                "source_index": detection.source_index,
                "grid": list(detection.grid),
                "stride": detection.stride,
            }
        )

    accelerated_identity = getattr(backend, "identity", None)
    if accelerated_identity is None:
        backend_record = {
            "name": "ambarella-ades-host-emulator",
            "container_image": runtime_spec.image,
            "platform": runtime_spec.platform,
            "cavalry_version": execution.execution.cavalry_version,
            "cavalry_hash": execution.execution.cavalry_hash,
        }
    else:
        backend_record = {
            "name": accelerated_identity["backend"],
            **dict(accelerated_identity),
        }

    document: dict[str, object] = {
        "schema": "verkeye.exact-image-inference.v1",
        "run_id": run_id,
        "model": dict(model_record or _artifact(model_path)),
        "image": image_record,
        "backend": backend_record,
        "preprocess": prepared.profile_manifest,
        "execution": {
            "split_count": len(execution.execution.splits),
            "wall_ms": execution.wall_ms,
            "phase_ns": {
                **dict(execution.timings_ns),
                **(
                    {"source_decode": source_decode_ns}
                    if source_decode_ns is not None
                    else {}
                ),
                "preprocess": preprocess_ns,
                "postprocess": postprocess_ns,
            },
            "split_duration_us": [
                item.duration_us for item in execution.execution.splits
            ],
        },
        "raw_tensors": tensor_records,
        "raw_prediction_matrix": {
            "shape": list(execution.predictions.shape),
            "dtype": str(execution.predictions.dtype),
            "sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        },
        "postprocess": {
            "scope": "binary-proved-core",
            "operating_profile": operating_profile,
            "profile": profile.to_manifest(),
            "class_labels": {
                str(class_id): label for class_id, label in _LABELS.items()
            },
            "claim_boundary": (
                "raw model execution and core detector decoding are exact; "
                "the later stateful cross-frame deduplication stage is not "
                "applied to this single-image result"
            ),
        },
        "detections": serialized_detections,
        "fidelity": {
            "exact_recovered_model": True,
            "raw_tensor_preservation": True,
            "dvi_hash_gate": "passed before backend execution",
            "physical_cv22_parity": "pending independent camera fixture",
        },
    }
    return ExactFrameResult(document=document, detections=detections)


def _run_exact_decoded(
    *,
    model_path: Path,
    decoded: np.ndarray,
    image_record: dict[str, object],
    backend: ExactInferenceBackend,
    runtime_spec: AdesRuntimeSpec,
    profile: CB62RawHeadProfile,
    operating_profile: str,
    run_id: str,
    source_decode_ns: int | None = None,
    model_record: dict[str, object] | None = None,
) -> dict[str, object]:
    return _execute_exact_decoded(
        model_path=model_path,
        decoded=decoded,
        image_record=image_record,
        backend=backend,
        runtime_spec=runtime_spec,
        profile=profile,
        operating_profile=operating_profile,
        run_id=run_id,
        source_decode_ns=source_decode_ns,
        model_record=model_record,
    ).document


def _decoded_metadata(
    decoded: np.ndarray,
    *,
    source_kind: str,
    source_path: Path | None,
    frame_index: int,
    timestamp_seconds: float | None,
    capture_monotonic_ns: int | None = None,
) -> dict[str, object]:
    return {
        "source_kind": source_kind,
        "source_path": str(source_path.resolve()) if source_path is not None else None,
        "frame_index": frame_index,
        "source_timestamp_seconds": timestamp_seconds,
        "capture_monotonic_ns": capture_monotonic_ns,
        "width": int(decoded.shape[1]),
        "height": int(decoded.shape[0]),
        "decoded_dtype": str(decoded.dtype),
        "decoded_bgr_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
    }


class ExactFrameSession:
    """Verify exact assets once and execute decoded media frames repeatedly."""

    def __init__(
        self,
        *,
        model: str | Path,
        backend: ExactInferenceBackend,
        runtime_spec: AdesRuntimeSpec,
        pipeline_evidence: str | Path,
        operating_profile: str = "production",
    ) -> None:
        self.model_path = Path(model)
        self.backend = backend
        self.runtime_spec = runtime_spec
        self.operating_profile = operating_profile
        runtime_spec.verify_model(self.model_path)
        self.profile = load_cb62_core_profile(
            pipeline_evidence,
            expected_sha256=runtime_spec.pipeline_evidence_sha256,
            operating_profile=operating_profile,
        )
        self.model_record = _artifact(self.model_path)

    def infer_frame(self, frame: MediaFrame, *, run_id: str) -> ExactFrameResult:
        """Execute one already-decoded exact-boundary frame."""

        image_record = _decoded_metadata(
            frame.image,
            source_kind=frame.source_kind,
            source_path=frame.path,
            frame_index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            capture_monotonic_ns=frame.capture_monotonic_ns,
        )
        return _execute_exact_decoded(
            model_path=self.model_path,
            decoded=frame.image,
            image_record=image_record,
            backend=self.backend,
            runtime_spec=self.runtime_spec,
            profile=self.profile,
            operating_profile=self.operating_profile,
            run_id=run_id,
            source_decode_ns=frame.decode_elapsed_ns,
            model_record=self.model_record,
        )

    def infer_live_frame(
        self,
        frame: MediaFrame,
        *,
        run_id: str,
    ) -> ExactLiveFrameResult:
        """Execute the identical exact core without per-frame forensic JSON."""

        preprocess_started = time.monotonic_ns()
        try:
            prepared = preprocess_cb62_exact_bgr(frame.image)
        except ValueError as error:
            raise ExactImageInferenceError(str(error)) from error
        preprocess_ns = time.monotonic_ns() - preprocess_started
        execution = self.backend.infer(prepared.tensor.tobytes(), run_id=run_id)
        postprocess_started = time.monotonic_ns()
        detections = decode_cb62_raw_head_core(execution.predictions, self.profile)
        postprocess_ns = time.monotonic_ns() - postprocess_started

        accelerated_identity = getattr(self.backend, "identity", None)
        if accelerated_identity is None:
            backend_record: dict[str, object] = {
                "name": "ambarella-ades-host-emulator",
                "container_image": self.runtime_spec.image,
                "platform": self.runtime_spec.platform,
                "cavalry_version": execution.execution.cavalry_version,
                "cavalry_hash": execution.execution.cavalry_hash,
            }
        else:
            backend_record = {
                "name": accelerated_identity["backend"],
                **dict(accelerated_identity),
            }
        return ExactLiveFrameResult(
            run_id=run_id,
            detections=detections,
            backend=backend_record,
            fidelity={
                "exact_recovered_model": True,
                "postprocess_operating_profile": self.operating_profile,
                "dvi_hash_gate": "passed before backend execution",
                "physical_cv22_parity": "pending independent camera fixture",
            },
            timings_ns={
                **dict(execution.timings_ns),
                "preprocess": preprocess_ns,
                "postprocess": postprocess_ns,
            },
        )


def run_exact_image(
    *,
    model: str | Path,
    image: str | Path,
    backend: ExactInferenceBackend,
    runtime_spec: AdesRuntimeSpec,
    pipeline_evidence: str | Path,
    run_id: str,
    operating_profile: str = "production",
) -> dict[str, object]:
    """Decode one exact-size image, execute ADES, and preserve all evidence."""

    import cv2

    model_path = Path(model)
    image_path = Path(image)
    session = ExactFrameSession(
        model=model_path,
        backend=backend,
        runtime_spec=runtime_spec,
        pipeline_evidence=pipeline_evidence,
        operating_profile=operating_profile,
    )
    decode_started = time.monotonic_ns()
    decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    decode_elapsed_ns = time.monotonic_ns() - decode_started
    if decoded is None:
        raise ExactImageInferenceError(f"cannot decode image {image_path}")
    image_record = {
        **_artifact(image_path),
        **_decoded_metadata(
            decoded,
            source_kind="image",
            source_path=image_path,
            frame_index=0,
            timestamp_seconds=None,
        ),
    }
    return _run_exact_decoded(
        model_path=model_path,
        decoded=decoded,
        image_record=image_record,
        backend=backend,
        runtime_spec=runtime_spec,
        profile=session.profile,
        operating_profile=session.operating_profile,
        run_id=run_id,
        source_decode_ns=decode_elapsed_ns,
        model_record=session.model_record,
    )


def run_exact_stream(
    *,
    model: str | Path,
    frames: Iterable[MediaFrame],
    backend: ExactInferenceBackend,
    runtime_spec: AdesRuntimeSpec,
    pipeline_evidence: str | Path,
    run_id: str,
    operating_profile: str = "production",
) -> dict[str, object]:
    """Execute an ordered finite or live frame iterable through the exact core."""

    session = ExactFrameSession(
        model=model,
        backend=backend,
        runtime_spec=runtime_spec,
        pipeline_evidence=pipeline_evidence,
        operating_profile=operating_profile,
    )
    documents: list[dict[str, object]] = []
    source_kind: str | None = None
    frame_indices: list[int] = []
    timestamps: list[float | None] = []
    total_wall_ms = 0
    try:
        for frame in frames:
            if source_kind is None:
                source_kind = frame.source_kind
            elif frame.source_kind != source_kind:
                raise ExactImageInferenceError(
                    "stream source kind changed between frames"
                )
            frame_run_id = f"{run_id}-f{frame.index:06d}"
            document = session.infer_frame(frame, run_id=frame_run_id).document
            documents.append(document)
            frame_indices.append(frame.index)
            timestamps.append(frame.timestamp_seconds)
            total_wall_ms += int(document["execution"]["wall_ms"])  # type: ignore[index]
    finally:
        close = getattr(frames, "close", None)
        if callable(close):
            close()
    if not documents or source_kind is None:
        raise ExactImageInferenceError("media source produced no frames")
    frame_count = len(documents)
    result = {
        "schema": "verkeye.exact-stream-inference.v1",
        "run_id": run_id,
        "source_kind": source_kind,
        "frame_count": frame_count,
        "frame_indices": frame_indices,
        "source_timestamps_seconds": timestamps,
        "execution": {
            "total_backend_wall_ms": total_wall_ms,
            "mean_backend_wall_ms": total_wall_ms / frame_count,
            "effective_backend_fps": (
                frame_count * 1000.0 / total_wall_ms
                if total_wall_ms > 0
                else None
            ),
        },
        "frames": documents,
        "fidelity": {
            "exact_recovered_model": True,
            "raw_tensor_preservation_per_frame": True,
            "physical_cv22_parity": "pending independent camera fixture",
            "realtime_claim": False,
        },
    }
    capture_stats = getattr(frames, "stats_snapshot", None)
    if callable(capture_stats):
        result["capture"] = capture_stats()
    return result
