#!/usr/bin/env python3
"""Capture deterministic ADES split-boundary oracle bundles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from verkeye.compat.ades_runtime import DockerAdesRuntime, load_runtime_spec
from verkeye.compat.oracle import load_oracle_bundle, write_oracle_bundle
from verkeye.runtime.inference import load_cb62_core_profile
from verkeye.runtime.preprocess import preprocess_cb62_exact_bgr
from verkeye.runtime.yolov6 import decode_cb62_raw_head_core


HEIGHT = 608
WIDTH = 1088


@dataclass(frozen=True, slots=True)
class OracleCase:
    input_tensor: bytes
    provenance: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--executor",
        type=Path,
        default=Path("src/verkeye/compat/native/ades_executor.cpp"),
    )
    parser.add_argument("--docker-command", nargs="+", default=["docker"])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="add an exact-size decoded BGR image fixture",
    )
    parser.add_argument(
        "--video-frame",
        action="append",
        default=[],
        metavar="NAME=PATH:INDEX",
        help="add one deterministically selected exact-size decoded video frame",
    )
    parser.add_argument(
        "--image-directory",
        action="append",
        default=[],
        metavar="PREFIX=PATH",
        help="add all sorted PNG images in one public-camera directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_runtime_spec(args.runtime)
    runtime = DockerAdesRuntime(
        spec=spec,
        model=args.model,
        workspace=args.workspace,
        executor_source=args.executor,
        docker_command=tuple(args.docker_command),
    )
    if args.prepare:
        runtime.prepare()
    else:
        runtime.prepared()
    profile = load_cb62_core_profile(
        args.pipeline, expected_sha256=spec.pipeline_evidence_sha256
    )
    cases = _standard_cases()
    synthetic = Path(".runtime/cb62-testsrc-1088x608.png")
    if synthetic.is_file():
        cases["synthetic-media"] = _image_case(
            synthetic,
            kind="generated-image",
        )
    for raw in args.image:
        name, separator, value = raw.partition("=")
        if not separator or not name or not value:
            raise SystemExit(f"invalid --image value: {raw!r}; expected NAME=PATH")
        if name in cases:
            raise SystemExit(f"duplicate oracle case: {name}")
        image = Path(value)
        kind = (
            "public-traffic-camera"
            if "public-webcams" in image.parts
            else "external-image"
        )
        cases[name] = _image_case(image, kind=kind)
    for raw in args.image_directory:
        prefix, separator, value = raw.partition("=")
        directory = Path(value)
        if not separator or not prefix or not value or not directory.is_dir():
            raise SystemExit(
                f"invalid --image-directory value: {raw!r}; expected PREFIX=PATH"
            )
        images = sorted(directory.glob("*.png"))
        if not images:
            raise SystemExit(f"image directory contains no PNG files: {directory}")
        for image in images:
            name = f"{prefix}-{image.stem}"
            if name in cases:
                raise SystemExit(f"duplicate oracle case: {name}")
            cases[name] = _image_case(image, kind="public-traffic-camera")
    for raw in args.video_frame:
        name, separator, value = raw.partition("=")
        path_text, index_separator, index_text = value.rpartition(":")
        if (
            not separator
            or not index_separator
            or not name
            or not path_text
            or not index_text.isascii()
            or not index_text.isdecimal()
        ):
            raise SystemExit(
                f"invalid --video-frame value: {raw!r}; expected NAME=PATH:INDEX"
            )
        if name in cases:
            raise SystemExit(f"duplicate oracle case: {name}")
        cases[name] = _video_frame_case(Path(path_text), int(index_text))

    args.output.mkdir(parents=True, exist_ok=True)
    catalog_cases: list[dict[str, object]] = []
    for case_id, case in cases.items():
        destination = args.output / case_id
        if destination.exists() and args.skip_existing:
            bundle = load_oracle_bundle(destination)
        else:
            inference = runtime.infer(
                case.input_tensor,
                run_id=f"oracle-{case_id}",
            )
            detections = decode_cb62_raw_head_core(inference.predictions, profile)
            raw_tensors = {
                name: (inference.output_directory / name).read_bytes()
                for name in inference.tensor_sha256
            }
            write_oracle_bundle(
                destination,
                case_id=case_id,
                input_tensor=case.input_tensor,
                raw_tensors=raw_tensors,
                predictions=inference.predictions,
                detections=tuple(
                    {
                        "class_id": item.class_id,
                        "confidence": item.confidence,
                        "normalized_box": list(item.normalized_box),
                        "source_index": item.source_index,
                        "grid": list(item.grid),
                        "stride": item.stride,
                    }
                    for item in detections
                ),
                model_sha256=spec.model_sha256,
                pipeline_sha256=spec.pipeline_evidence_sha256,
                runtime_identity={
                    "name": "ambarella-ades-host-emulator",
                    "container_image": spec.image,
                    "platform": spec.platform,
                    "cavalry_version": spec.cavalry_version,
                    "cavalry_hash": spec.cavalry_hash,
                },
            )
            bundle = load_oracle_bundle(destination)
        catalog_cases.append(
            {
                "case_id": bundle.case_id,
                "bundle": destination.as_posix(),
                "manifest_sha256": bundle.manifest_sha256,
                "input_sha256": hashlib.sha256(bundle.input_tensor).hexdigest(),
                "prediction_sha256": hashlib.sha256(
                    bundle.predictions.tobytes(order="C")
                ).hexdigest(),
                "detection_count": len(bundle.detections),
                "raw_tensor_count": len(bundle.raw_tensors),
                "provenance": case.provenance,
            }
        )
        print(f"captured {case_id}: {bundle.manifest_sha256}", flush=True)

    catalog = {
        "schema": "verkeye.ades-oracle-catalog.v2",
        "model": {
            "path": args.model.as_posix(),
            "sha256": spec.model_sha256,
        },
        "pipeline": {
            "path": args.pipeline.as_posix(),
            "sha256": spec.pipeline_evidence_sha256,
        },
        "oracle": {
            "name": "ambarella-ades-host-emulator",
            "container_image": spec.image,
            "platform": spec.platform,
            "cavalry_version": spec.cavalry_version,
            "cavalry_hash": spec.cavalry_hash,
        },
        "case_count": len(catalog_cases),
        "cases": catalog_cases,
    }
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def _standard_cases() -> dict[str, OracleCase]:
    zero = np.zeros((1, 3, HEIGHT, WIDTH), dtype=np.uint8)
    impulse = zero.copy()
    impulse[:, :, HEIGHT // 2, WIDTH // 2] = 255
    plane = (np.arange(HEIGHT * WIDTH, dtype=np.uint32) % 256).astype(np.uint8)
    ramp = np.stack((plane, np.roll(plane, 1), np.roll(plane, 2))).reshape(
        1, 3, HEIGHT, WIDTH
    )
    random = np.random.default_rng(0xCB62).integers(
        0, 256, size=(1, 3, HEIGHT, WIDTH), dtype=np.uint8
    )
    checkerboard = np.indices((HEIGHT, WIDTH)).sum(axis=0) % 2
    checkerboard = np.broadcast_to(
        (checkerboard * 255).astype(np.uint8),
        (1, 3, HEIGHT, WIDTH),
    ).copy()
    boundaries = np.asarray([0, 1, 127, 128, 254, 255], dtype=np.uint8)
    boundary_pattern = np.resize(
        boundaries,
        (1, 3, HEIGHT, WIDTH),
    )

    def generated(
        tensor: np.ndarray,
        *,
        kind: str,
        generator: str,
    ) -> OracleCase:
        return OracleCase(
            np.ascontiguousarray(tensor).tobytes(order="C"),
            {
                "kind": kind,
                "generator": generator,
                "shape": [1, 3, HEIGHT, WIDTH],
                "dtype": "uint8",
            },
        )

    return {
        "zero": generated(
            zero,
            kind="synthetic-constant",
            generator="numpy.full(value=0)",
        ),
        "constant-127": generated(
            np.full_like(zero, 127),
            kind="synthetic-constant",
            generator="numpy.full(value=127)",
        ),
        "constant-128": generated(
            np.full_like(zero, 128),
            kind="synthetic-constant",
            generator="numpy.full(value=128)",
        ),
        "boundary-six-values": generated(
            boundary_pattern,
            kind="synthetic-boundary",
            generator="numpy.resize(values=[0,1,127,128,254,255])",
        ),
        "impulse": generated(
            impulse,
            kind="synthetic-boundary",
            generator="zero tensor with all-channel center impulse=255",
        ),
        "ramp": generated(
            ramp,
            kind="synthetic-structured",
            generator="three planar modulo-256 ramps with offsets [0,1,2]",
        ),
        "checkerboard": generated(
            checkerboard,
            kind="synthetic-structured",
            generator="one-pixel checkerboard values [0,255]",
        ),
        "random-seed-cb62": generated(
            random,
            kind="synthetic-seeded-random",
            generator="numpy.default_rng(seed=0xCB62).integers(0,256)",
        ),
    }


def _image_case(path: Path, *, kind: str) -> OracleCase:
    import cv2

    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None:
        raise SystemExit(f"cannot decode oracle image: {path}")
    try:
        tensor = preprocess_cb62_exact_bgr(decoded).tensor.tobytes(order="C")
    except ValueError as exc:
        raise SystemExit(f"invalid oracle image {path}: {exc}") from exc
    return OracleCase(
        tensor,
        {
            "kind": kind,
            "source": path.as_posix(),
            "source_file_sha256": _sha256(path),
            "decoded_bgr_sha256": hashlib.sha256(
                decoded.tobytes(order="C")
            ).hexdigest(),
            "decoder": f"opencv-{cv2.__version__}",
            "frame_index": 0,
        },
    )


def _video_frame_case(path: Path, frame_index: int) -> OracleCase:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise SystemExit(f"cannot open oracle video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, decoded = capture.read()
    finally:
        capture.release()
    if not ok or decoded is None:
        raise SystemExit(f"cannot decode oracle video frame {frame_index}: {path}")
    try:
        tensor = preprocess_cb62_exact_bgr(decoded).tensor.tobytes(order="C")
    except ValueError as exc:
        raise SystemExit(
            f"invalid oracle video frame {frame_index} from {path}: {exc}"
        ) from exc
    return OracleCase(
        tensor,
        {
            "kind": "decoded-video-frame",
            "source": path.as_posix(),
            "source_file_sha256": _sha256(path),
            "decoded_bgr_sha256": hashlib.sha256(
                decoded.tobytes(order="C")
            ).hexdigest(),
            "decoder": f"opencv-{cv2.__version__}",
            "frame_index": frame_index,
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
