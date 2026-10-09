"""Command-line entry point for evidence-first VerkEye operations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import sysconfig
from itertools import islice
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from .accelerated.cb62 import (
    AcceleratedCb62Runtime,
    Cb62AcceleratedError,
    Cb62MlxSession,
)
from .accelerated.openvino import OpenVinoCb62Session
from .compat.ades_runtime import (
    AdesRuntimeError,
    DockerAdesRuntime,
    load_runtime_spec,
)
from .cv22.container import inspect_path
from .cv22.graph import inspect_tensor_graph_path
from .cv22.weights import inspect_weight_map_path
from .evidence import (
    artifact_record,
    atomic_write_json,
    atomic_write_text,
    execution_manifest,
)
from .ir import build_ir_path
from .ir_json import dumps_ir, ir_to_document
from .runtime.inference import (
    ExactFrameSession,
    ExactImageInferenceError,
    run_exact_image,
    run_exact_stream,
)
from .runtime.generation import (
    GeneratedRuntimeError,
    generated_runtime_path,
    generate_owner_runtime,
    load_generated_runtime,
)
from .runtime.live import LiveCaptureStream
from .runtime.sources import MediaSourceError, iter_capture, iter_frame_directory
from .viewer.controller import (
    LiveViewerSession,
    OpenCvDisplay,
    ViewerError,
    open_video_writer,
)


def _default_runtime_asset(relative: str) -> Path:
    """Resolve a development-tree or wheel-installed immutable runtime asset."""

    source_relative = Path(relative)
    repository_asset = Path(__file__).resolve().parents[2] / source_relative
    installed_asset = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "verkeye"
        / "ades"
        / source_relative.name
    )
    candidates = (
        repository_asset,
        installed_asset,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return installed_asset


def _json_command(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
) -> argparse.ArgumentParser:
    command = subcommands.add_parser(name, help=help_text)
    command.add_argument("model", type=Path)
    command.add_argument("--json-out", type=Path, required=True)
    command.add_argument("--manifest-out", type=Path)
    return command


def _add_exact_backend_arguments(
    command: argparse.ArgumentParser,
    *,
    default_backend: str,
) -> None:
    command.add_argument(
        "--backend",
        choices=("ades", "accelerated", "auto"),
        default=default_backend,
    )
    command.add_argument(
        "--workspace", type=Path, default=Path(".runtime/ades-cb62")
    )
    command.add_argument(
        "--runtime-spec",
        type=Path,
        default=_default_runtime_asset("config/cb62-ades-runtime.json"),
    )
    command.add_argument(
        "--pipeline-evidence",
        type=Path,
        default=_default_runtime_asset(
            "evidence/compatibility/cvproc-yolov6-pipeline.json"
        ),
    )
    command.add_argument(
        "--executor-source",
        type=Path,
        default=_default_runtime_asset(
            "src/verkeye/compat/native/ades_executor.cpp"
        ),
    )
    command.add_argument(
        "--generated-runtime-base",
        type=Path,
        default=Path(".runtime/generated"),
        help="content-addressed owner-local runtime directory",
    )
    command.add_argument(
        "--accelerated-capture-root",
        type=Path,
        default=None,
    )
    command.add_argument(
        "--accelerated-split4-root",
        type=Path,
        default=None,
    )
    command.add_argument(
        "--accelerated-split5-root",
        type=Path,
        default=None,
    )
    command.add_argument(
        "--accelerated-device",
        choices=("auto", "CPU", "GPU"),
        default="auto",
        help="accelerator device for the OpenVINO provider (default: auto)",
    )
    command.add_argument(
        "--prepare",
        action="store_true",
        help="rebuild and reverify the digest-pinned ADES workspace",
    )


def _add_detection_profile_argument(
    command: argparse.ArgumentParser,
    *,
    default: str,
) -> None:
    command.add_argument(
        "--detection-profile",
        choices=("production", "small-object"),
        default=default,
        help=(
            "host-side detector filtering profile; small-object preserves the "
            "exact recovered model and lowers only the minimum box-area gate"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verkeye")
    parser.add_argument(
        "--demo",
        type=int,
        choices=(1, 2),
        help="run a zero-configuration GUI demonstration",
    )
    subcommands = parser.add_subparsers(dest="command", required=False)
    _json_command(subcommands, "inspect", "map an exact CV22 model container")
    _json_command(
        subcommands,
        "map",
        "emit the complete recovered CV22 structure and feasibility gates",
    )
    _json_command(
        subcommands,
        "ir",
        "emit the provenance-preserving VerkEye IR v1 document",
    )

    convert = subcommands.add_parser(
        "convert", help="export only after all exact-conversion gates pass"
    )
    convert.add_argument("model", type=Path)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--manifest-out", type=Path, required=True)

    validate = subcommands.add_parser(
        "validate", help="validate only after an exact export exists"
    )
    validate.add_argument("model", type=Path)
    validate.add_argument("converted", type=Path)
    validate.add_argument("--manifest-out", type=Path, required=True)

    generate = subcommands.add_parser(
        "generate-runtime",
        help="derive accelerated runtime assets locally from an owner model",
    )
    generate.add_argument("model", type=Path)
    generate.add_argument(
        "--output", type=Path, default=Path(".runtime/generated")
    )
    generate.add_argument(
        "--runtime-spec",
        type=Path,
        default=_default_runtime_asset("config/cb62-ades-runtime.json"),
    )
    generate.add_argument(
        "--executor-source",
        type=Path,
        default=_default_runtime_asset(
            "src/verkeye/compat/native/ades_executor.cpp"
        ),
    )
    generate.add_argument(
        "--kernel-capture-source",
        type=Path,
        default=_default_runtime_asset(
            "src/verkeye/compat/native/ades_kernel_capture.cpp"
        ),
    )
    generate.add_argument(
        "--operator-capture-source",
        type=Path,
        default=_default_runtime_asset(
            "src/verkeye/compat/native/ades_operator_capture.cpp"
        ),
    )
    generate.add_argument("--workspace", type=Path)
    generate.add_argument(
        "--docker-command",
        nargs="+",
        default=["docker"],
        help="container command and optional fixed arguments (default: docker)",
    )
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--keep-workspace", action="store_true")

    run = subcommands.add_parser("run", help="execute the exact recovered model")
    run.add_argument("model", type=Path)
    sources = run.add_mutually_exclusive_group(required=True)
    sources.add_argument("--image", type=Path)
    sources.add_argument("--frames", type=Path)
    sources.add_argument("--video", type=Path)
    sources.add_argument("--webcam", type=int)
    _add_exact_backend_arguments(run, default_backend="ades")
    _add_detection_profile_argument(run, default="production")
    run.add_argument("--json-out", type=Path, required=True)
    run.add_argument("--manifest-out", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument(
        "--max-frames",
        type=int,
        help=(
            "stop after this many frames; required for webcam input and "
            "optional for video or frame directories"
        ),
    )
    run.add_argument(
        "--queue-size",
        type=int,
        default=2,
        help="bounded drop-oldest capture queue size for webcam input (default: 2)",
    )
    live = subcommands.add_parser(
        "live",
        help="view exact recovered CB62 detections from a webcam or video",
    )
    live.add_argument("model", type=Path)
    live_sources = live.add_mutually_exclusive_group(required=True)
    live_sources.add_argument(
        "--webcam", type=int, help="local camera index, such as 0 or 1"
    )
    live_sources.add_argument("--video", type=Path)
    _add_exact_backend_arguments(live, default_backend="auto")
    _add_detection_profile_argument(live, default="small-object")
    live.add_argument("--run-id")
    live.add_argument(
        "--queue-size",
        type=int,
        default=2,
        help="bounded drop-oldest webcam queue capacity (default: 2)",
    )
    live.add_argument(
        "--max-frames",
        type=int,
        help="optional exact-inference frame limit for a bounded session",
    )
    live.add_argument("--record", type=Path, help="write annotated video")
    live.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="annotated recording rate (default: 30)",
    )
    live.add_argument(
        "--record-codec",
        default="mp4v",
        help="four-character OpenCV recording codec (default: mp4v)",
    )
    live.add_argument(
        "--summary-out",
        type=Path,
        default=Path("verkeye-live-summary.json"),
    )
    live.add_argument(
        "--window-title",
        default="VerkEye — exact CB62 live viewer",
    )
    return parser


def _run_demo_command(demo: int) -> int:
    from .demo import run_demo

    return run_demo(demo)


def _generate_runtime_command(args: argparse.Namespace) -> int:
    spec = load_runtime_spec(args.runtime_spec)
    workspace = args.workspace
    if args.keep_workspace and workspace is None:
        workspace = (
            args.output.parent
            / f"generation-workspace-{spec.model_sha256[:12]}"
        )
    runtime = generate_owner_runtime(
        model=args.model,
        output_base=args.output,
        spec=spec,
        executor_source=args.executor_source,
        kernel_capture_source=args.kernel_capture_source,
        operator_capture_source=args.operator_capture_source,
        workspace=workspace,
        docker_command=tuple(args.docker_command),
        force=args.force,
        keep_workspace=args.keep_workspace,
    )
    summary = {
        "schema": "verkeye.runtime-generation-result.v1",
        "runtime_root": str(runtime.root.resolve()),
        "model_sha256": runtime.model_sha256,
        "fastconv_count": len(runtime.fastconv),
        "mask_members": [
            str(runtime.split4_mask.relative_to(runtime.root)),
            str(runtime.split5_mask.relative_to(runtime.root)),
        ],
    }
    if workspace is not None:
        summary["workspace"] = str(workspace.resolve())
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _gate_summary(
    tensor_document: dict[str, Any], weight_document: dict[str, Any]
) -> dict[str, str]:
    return {
        "container": "passed",
        "tensor": tensor_document["gate"]["status"],
        "weight": weight_document["weight_gate"]["status"],
        "quantization": weight_document["quantization_gate"]["status"],
        "onnx": "blocked",
        "inference": "blocked",
    }


def _incomplete_dispositions(weight_document: dict[str, Any]) -> dict[str, int]:
    return {
        "opaque_compiled_bytes": weight_document["coverage"][
            "opaque_compiled_bytes"
        ],
        "semantic_parameter_bytes": weight_document["coverage"][
            "semantic_parameter_bytes"
        ],
        "unknown_quantization_records": weight_document["quantization_gate"][
            "tensor_count"
        ],
        "unknown_split_operators": weight_document["coverage"][
            "compiled_block_count"
        ],
    }


def _recovery_map(model: Path) -> dict[str, Any]:
    container = inspect_path(model)
    tensors = inspect_tensor_graph_path(model)
    compiled = inspect_weight_map_path(model)
    gates = _gate_summary(tensors, compiled)
    return {
        "schema": "verkeye.cv22.recovery-map.v1",
        "artifact": container["artifact"],
        "gates": gates,
        "container": container,
        "tensor_graph": tensors,
        "compiled_package": compiled,
        "interpretation_boundary": (
            "container, boundary tensors, split edges, and compiled spans are "
            "mapped; internal operators, parameters, and numeric semantics "
            "remain unresolved"
        ),
    }


def _default_manifest_path(json_output: Path) -> Path:
    return Path(f"{json_output}.manifest.json")


def _json_text(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _write_success(
    *,
    args: argparse.Namespace,
    raw_argv: tuple[str, ...],
    content: str,
    gates: dict[str, str],
    warnings: tuple[str, ...],
    incomplete: dict[str, int],
) -> int:
    atomic_write_text(args.json_out, content)
    manifest_path = args.manifest_out or _default_manifest_path(args.json_out)
    manifest = execution_manifest(
        command=args.command,
        argv=("verkeye", *raw_argv),
        status="passed",
        inputs=(artifact_record(args.model),),
        outputs=(artifact_record(args.json_out),),
        gates=gates,
        warnings=warnings,
        incomplete_dispositions=incomplete,
    )
    atomic_write_json(manifest_path, manifest)
    print(f"wrote {args.json_out}")
    print(f"wrote {manifest_path}")
    return 0


def _blocked_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    paths = [args.model]
    if args.command == "validate":
        paths.append(args.converted)
    elif args.command == "run":
        for name in ("image", "frames", "video"):
            value = getattr(args, name)
            if value is not None:
                paths.append(value)
    return tuple(artifact_record(path) for path in paths)


def _write_blocked(
    args: argparse.Namespace, raw_argv: tuple[str, ...]
) -> int:
    tensors = inspect_tensor_graph_path(args.model)
    compiled = inspect_weight_map_path(args.model)
    gates = _gate_summary(tensors, compiled)
    blockers = (
        "weight_gate_blocked",
        "quantization_gate_blocked",
        "exact_executable_graph_unavailable",
    )
    manifest = execution_manifest(
        command=args.command,
        argv=("verkeye", *raw_argv),
        status="blocked",
        inputs=_blocked_inputs(args),
        outputs=(),
        gates=gates,
        warnings=(
            "No output was created because exact-conversion prerequisites failed.",
        ),
        incomplete_dispositions=_incomplete_dispositions(compiled),
        blockers=blockers,
    )
    atomic_write_json(args.manifest_out, manifest)
    print(f"wrote {args.manifest_out}")
    print(
        "blocked: exact model parameters and numeric semantics are unresolved",
        file=sys.stderr,
    )
    return 2


def _exact_backend(args: argparse.Namespace) -> tuple[Any, Any]:
    spec = load_runtime_spec(args.runtime_spec)
    mlx_available = importlib.util.find_spec("mlx") is not None
    openvino_available = importlib.util.find_spec("openvino") is not None
    legacy_assets_available = (
        args.accelerated_capture_root is not None
        and args.accelerated_split4_root is not None
        and args.accelerated_split5_root is not None
        and args.accelerated_capture_root.is_dir()
        and args.accelerated_split4_root.is_dir()
        and args.accelerated_split5_root.is_dir()
    )
    generated_root = generated_runtime_path(
        args.generated_runtime_base, spec.model_sha256
    )
    generated_available = (generated_root / "manifest.json").is_file()
    assets_available = legacy_assets_available or generated_available
    accelerated_available = assets_available and (
        mlx_available or openvino_available
    )
    if args.backend == "accelerated" and not assets_available:
        raise Cb62AcceleratedError(
            "the accelerated CB62 runtime is unavailable; run "
            "`verkeye generate-runtime OWNER_MODEL.bin` first"
        )
    if args.backend == "accelerated" and not (
        mlx_available or openvino_available
    ):
        raise Cb62AcceleratedError(
            "neither the MLX nor OpenVINO accelerated provider is installed"
        )
    if args.backend == "accelerated" or (
        args.backend == "auto" and accelerated_available
    ):
        if legacy_assets_available:
            session_kwargs: dict[str, object] = {
                "capture_root": args.accelerated_capture_root,
                "split4_root": args.accelerated_split4_root,
                "split5_root": args.accelerated_split5_root,
            }
        else:
            session_kwargs = {
                "generated_runtime": load_generated_runtime(
                    generated_root,
                    expected_model_sha256=spec.model_sha256,
                )
            }
        if mlx_available:
            session = Cb62MlxSession(**session_kwargs)
        else:
            import openvino as ov

            devices = tuple(ov.Core().available_devices)
            requested_device = args.accelerated_device
            if requested_device == "auto":
                device = "GPU" if "GPU" in devices else "CPU"
            else:
                device = requested_device
            if device not in devices:
                raise Cb62AcceleratedError(
                    f"OpenVINO device {device} is unavailable; found {devices}"
                )
            session = OpenVinoCb62Session(
                device=device,
                **session_kwargs,
            )
        return spec, AcceleratedCb62Runtime(session=session)
    backend = DockerAdesRuntime(
        spec=spec,
        model=args.model,
        workspace=args.workspace,
        executor_source=args.executor_source,
    )
    if args.prepare:
        backend.prepare()
    else:
        try:
            backend.prepared()
        except AdesRuntimeError:
            backend.prepare()
    return spec, backend


def _exact_manifest_inputs(
    args: argparse.Namespace, source: Path | None
) -> tuple[dict[str, Any], ...]:
    paths = [
        args.model,
        args.runtime_spec,
        args.pipeline_evidence,
        args.executor_source,
    ]
    if source is not None:
        paths.insert(1, source)
    return tuple(artifact_record(path) for path in paths)


def _write_exact_result(
    *,
    args: argparse.Namespace,
    raw_argv: tuple[str, ...],
    document: dict[str, Any],
    source: Path | None,
    stream: bool,
) -> int:
    atomic_write_json(args.json_out, document)
    warnings = [
        "Physical CV22 parity awaits an independent camera fixture.",
    ]
    incomplete = {"physical_cv22_parity_fixtures": 1}
    gates = {
        "exact_model_execution": "passed",
        "raw_tensor_preservation": "passed",
        "binary_proved_postprocess_core": "passed",
        "physical_cv22_parity": "pending",
    }
    if stream:
        gates["realtime_throughput"] = "not_claimed"
        warnings.extend(
            (
                "The measured host-emulator throughput is reported; real-time "
                "performance is not claimed.",
                "The unresolved stateful cross-frame deduplication stage is "
                "not applied.",
            )
        )
        incomplete["stateful_cross_frame_dedup_stages"] = 1
    else:
        warnings.append(
            "Stateful cross-frame deduplication is not applied to one image."
        )
        incomplete["stateful_cross_frame_dedup_runs"] = 1
    manifest = execution_manifest(
        command=args.command,
        argv=("verkeye", *raw_argv),
        status="passed",
        inputs=_exact_manifest_inputs(args, source),
        outputs=(artifact_record(args.json_out),),
        gates=gates,
        warnings=tuple(warnings),
        incomplete_dispositions=incomplete,
    )
    atomic_write_json(args.manifest_out, manifest)
    print(f"wrote {args.json_out}")
    print(f"wrote {args.manifest_out}")
    return 0


def _run_exact_image_command(
    args: argparse.Namespace, raw_argv: tuple[str, ...]
) -> int:
    spec, backend = _exact_backend(args)
    run_id = args.run_id or f"image-{uuid4().hex[:16]}"
    document = run_exact_image(
        model=args.model,
        image=args.image,
        backend=backend,
        runtime_spec=spec,
        pipeline_evidence=args.pipeline_evidence,
        run_id=run_id,
        operating_profile=args.detection_profile,
    )
    return _write_exact_result(
        args=args,
        raw_argv=raw_argv,
        document=document,
        source=args.image,
        stream=False,
    )


def _run_exact_stream_command(
    args: argparse.Namespace, raw_argv: tuple[str, ...]
) -> int:
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.queue_size <= 0:
        raise ValueError("--queue-size must be positive")
    if args.webcam is not None and args.max_frames is None:
        raise ValueError("--webcam requires --max-frames")
    spec, backend = _exact_backend(args)
    if args.frames is not None:
        source: Path | None = args.frames
        frames = iter_frame_directory(args.frames)
        if args.max_frames is not None:
            frames = islice(frames, args.max_frames)
        run_prefix = "frames"
    elif args.video is not None:
        source = args.video
        frames = iter_capture(args.video, max_frames=args.max_frames)
        run_prefix = "video"
    elif args.webcam is not None:
        source = None
        frames = LiveCaptureStream(
            args.webcam,
            queue_capacity=args.queue_size,
            max_frames=args.max_frames,
        )
        run_prefix = "webcam"
    else:
        raise AssertionError("stream command has no media source")
    run_id = args.run_id or f"{run_prefix}-{uuid4().hex[:16]}"
    document = run_exact_stream(
        model=args.model,
        frames=frames,
        backend=backend,
        runtime_spec=spec,
        pipeline_evidence=args.pipeline_evidence,
        run_id=run_id,
        operating_profile=args.detection_profile,
    )
    return _write_exact_result(
        args=args,
        raw_argv=raw_argv,
        document=document,
        source=source,
        stream=True,
    )


def _run_live_command(args: argparse.Namespace) -> int:
    if args.queue_size <= 0:
        raise ValueError("--queue-size must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.record_fps <= 0:
        raise ValueError("--record-fps must be positive")
    if len(args.record_codec) != 4:
        raise ValueError("--record-codec must contain four characters")

    spec, backend = _exact_backend(args)
    exact_session = ExactFrameSession(
        model=args.model,
        backend=backend,
        runtime_spec=spec,
        pipeline_evidence=args.pipeline_evidence,
        operating_profile=args.detection_profile,
    )
    if args.webcam is not None:
        source = LiveCaptureStream(
            args.webcam,
            queue_capacity=args.queue_size,
            max_frames=None,
        )
        source_label = f"webcam {args.webcam}"
        run_prefix = "live-webcam"
    else:
        source = iter_capture(args.video)
        source_label = f"video {args.video}"
        run_prefix = "live-video"

    writer = None
    recording = None
    if args.record is not None:
        writer = open_video_writer(
            args.record,
            fps=args.record_fps,
            codec=args.record_codec,
        )
        recording = {
            "path": str(args.record.resolve()),
            "codec": args.record_codec,
            "fps": args.record_fps,
            "size": [608, 1088],
        }
    run_id = args.run_id or f"{run_prefix}-{uuid4().hex[:16]}"
    viewer = LiveViewerSession(
        source=source,
        exact_session=exact_session,
        display=OpenCvDisplay(),
        source_label=source_label,
        run_id=run_id,
        writer=writer,
        recording=recording,
        max_frames=args.max_frames,
        window_title=args.window_title,
    )
    summary = viewer.run()
    atomic_write_json(args.summary_out, summary)
    print(f"wrote {args.summary_out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(raw_argv)
    if args.demo is None and args.command is None:
        parser.error("a subcommand or --demo is required")
    if args.demo is not None and args.command is not None:
        parser.error("--demo cannot be combined with a subcommand")
    try:
        if args.demo is not None:
            return _run_demo_command(args.demo)
        if args.command == "inspect":
            document = inspect_path(args.model)
            return _write_success(
                args=args,
                raw_argv=raw_argv,
                content=_json_text(document),
                gates={"container": "passed"},
                warnings=(document["container"]["interpretation_boundary"],),
                incomplete={
                    "opaque_payload_bytes": document["container"][
                        "opaque_payload_bytes"
                    ],
                    "top_level_unknown_bytes": document["container"][
                        "top_level_unknown_bytes"
                    ],
                },
            )
        if args.command == "map":
            document = _recovery_map(args.model)
            compiled = document["compiled_package"]
            return _write_success(
                args=args,
                raw_argv=raw_argv,
                content=_json_text(document),
                gates=document["gates"],
                warnings=("Exact conversion is blocked",),
                incomplete=_incomplete_dispositions(compiled),
            )
        if args.command == "ir":
            ir = build_ir_path(args.model)
            document = ir_to_document(ir)
            return _write_success(
                args=args,
                raw_argv=raw_argv,
                content=dumps_ir(ir),
                gates={item.name: item.status for item in ir.gates},
                warnings=(
                    "Opaque split nodes preserve bytes but are not executable "
                    "neural-network operators.",
                ),
                incomplete={
                    "unknown_split_operators": len(document["graph"]["nodes"]),
                    "unknown_quantization_records": len(
                        document["graph"]["tensors"]
                    ),
                    "semantic_constants": len(document["graph"]["constants"]),
                },
            )
        if args.command == "generate-runtime":
            return _generate_runtime_command(args)
        if args.command == "run" and args.image is not None:
            return _run_exact_image_command(args, raw_argv)
        if args.command == "run":
            return _run_exact_stream_command(args, raw_argv)
        if args.command == "live":
            return _run_live_command(args)
        if args.command in {"convert", "validate", "run"}:
            return _write_blocked(args, raw_argv)
        raise AssertionError(f"unhandled command: {args.command}")
    except (
        OSError,
        ValueError,
        AdesRuntimeError,
        Cb62AcceleratedError,
        ExactImageInferenceError,
        MediaSourceError,
        GeneratedRuntimeError,
        ViewerError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
