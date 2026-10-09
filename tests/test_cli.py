from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from verkeye.cli import _default_runtime_asset, _parser, main


MODEL = Path("fixtures/models/yolov6n_hor.bin")
MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"


@pytest.mark.parametrize("demo", (1, 2))
def test_top_level_demo_selector_does_not_require_a_subcommand(demo: int) -> None:
    args = _parser().parse_args(["--demo", str(demo)])

    assert args.demo == demo
    assert args.command is None


def test_top_level_demo_selector_rejects_unknown_presets() -> None:
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["--demo", "3"])

    assert error.value.code == 2


def test_existing_live_command_does_not_select_a_demo() -> None:
    args = _parser().parse_args(["live", str(MODEL), "--webcam", "0"])

    assert args.demo is None
    assert args.command == "live"


def test_generate_runtime_command_has_safe_owner_local_defaults() -> None:
    args = _parser().parse_args(["generate-runtime", "owner-model.bin"])

    assert args.command == "generate-runtime"
    assert args.output == Path(".runtime/generated")
    assert args.docker_command == ["docker"]
    assert args.force is False
    assert args.keep_workspace is False
    assert args.runtime_spec.is_file()
    assert args.executor_source.is_file()
    assert args.kernel_capture_source.is_file()
    assert args.operator_capture_source.is_file()


def test_generate_runtime_dispatches_and_prints_safe_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "owner-model.bin"
    model.write_bytes(b"owner model")
    output = tmp_path / "generated"
    observed: dict[str, object] = {}

    class FakeGenerated:
        root = output / MODEL_SHA256
        model_sha256 = MODEL_SHA256
        fastconv = tuple(range(55))
        split4_mask = root / "masks/split4.bin"
        split5_mask = root / "masks/split5.bin"

    monkeypatch.setattr(
        "verkeye.cli.load_runtime_spec",
        lambda path: types.SimpleNamespace(model_sha256=MODEL_SHA256),
    )

    def fake_generate(**kwargs: object) -> object:
        observed.update(kwargs)
        return FakeGenerated()

    monkeypatch.setattr("verkeye.cli.generate_owner_runtime", fake_generate)

    result = main(
        [
            "generate-runtime",
            str(model),
            "--output",
            str(output),
            "--docker-command",
            "podman",
        ]
    )

    assert result == 0
    assert observed["model"] == model
    assert observed["output_base"] == output
    assert observed["docker_command"] == ("podman",)
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == "verkeye.runtime-generation-result.v1"
    assert summary["fastconv_count"] == 55


def test_main_dispatches_demo_without_constructing_a_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []
    monkeypatch.setattr(
        "verkeye.cli._run_demo_command",
        lambda demo: observed.append(demo) or 0,
        raising=False,
    )

    assert main(["--demo", "1"]) == 0
    assert observed == [1]


def test_exact_runtime_defaults_resolve_to_shipped_assets() -> None:
    args = _parser().parse_args(
        [
            "run",
            str(MODEL),
            "--image",
            "frame.png",
            "--json-out",
            "result.json",
            "--manifest-out",
            "manifest.json",
        ]
    )

    assert args.runtime_spec.is_file()
    assert args.pipeline_evidence.is_file()
    assert args.executor_source.is_file()
    assert args.detection_profile == "production"


def test_live_defaults_to_small_object_detection_profile() -> None:
    args = _parser().parse_args(
        [
            "live",
            str(MODEL),
            "--webcam",
            "0",
        ]
    )

    assert args.detection_profile == "small-object"


def test_run_accepts_explicit_small_object_detection_profile() -> None:
    args = _parser().parse_args(
        [
            "run",
            str(MODEL),
            "--image",
            "frame.png",
            "--detection-profile",
            "small-object",
            "--json-out",
            "result.json",
            "--manifest-out",
            "manifest.json",
        ]
    )

    assert args.detection_profile == "small-object"


def test_default_runtime_assets_ignore_cwd_shadow_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = tmp_path / "config" / "cb62-ades-runtime.json"
    shadow.parent.mkdir()
    shadow.write_text("untrusted cwd shadow\n")
    monkeypatch.chdir(tmp_path)

    resolved = _default_runtime_asset("config/cb62-ades-runtime.json")

    assert resolved.is_absolute()
    assert resolved != shadow
    assert resolved.read_text() != "untrusted cwd shadow\n"


@pytest.mark.parametrize(
    ("command", "schema"),
    [
        ("inspect", "verkeye.cv22.container.v1"),
        ("map", "verkeye.cv22.recovery-map.v1"),
        ("ir", "verkeye.ir.v1"),
    ],
)
def test_successful_commands_write_atomic_json_and_reproducibility_manifests(
    command: str,
    schema: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / f"{command}.json"
    manifest = tmp_path / f"{command}.manifest.json"
    argv = [
        command,
        str(MODEL),
        "--json-out",
        str(output),
        "--manifest-out",
        str(manifest),
    ]

    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == f"wrote {output}\nwrote {manifest}\n"

    document = json.loads(output.read_text())
    record = json.loads(manifest.read_text())
    assert document["schema"] == schema
    assert record["status"] == "passed"
    assert record["command"] == {
        "name": command,
        "argv": ["verkeye", *argv],
    }
    assert record["inputs"][0]["sha256"] == MODEL_SHA256
    assert record["outputs"][0]["path"] == str(output.resolve())
    assert record["outputs"][0]["sha256"]
    assert record["tool"]["version"] == "0.1.0.dev0"


def test_map_reports_each_gate_and_incomplete_disposition(tmp_path: Path):
    output = tmp_path / "map.json"
    manifest = tmp_path / "map.manifest.json"
    assert (
        main(
            [
                "map",
                str(MODEL),
                "--json-out",
                str(output),
                "--manifest-out",
                str(manifest),
            ]
        )
        == 0
    )

    document = json.loads(output.read_text())
    record = json.loads(manifest.read_text())
    assert document["gates"] == {
        "container": "passed",
        "tensor": "passed",
        "weight": "blocked",
        "quantization": "blocked",
        "onnx": "blocked",
        "inference": "blocked",
    }
    assert record["gates"] == document["gates"]
    assert record["incomplete_dispositions"] == {
        "opaque_compiled_bytes": 5_514_624,
        "semantic_parameter_bytes": 0,
        "unknown_quantization_records": 39,
        "unknown_split_operators": 10,
    }
    assert "Exact conversion is blocked" in record["warnings"]


@pytest.mark.parametrize("command", ["convert", "validate"])
def test_unavailable_execution_commands_fail_closed_with_structured_manifest(
    command: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    manifest = tmp_path / f"{command}.manifest.json"
    output = tmp_path / "model.onnx"
    converted = tmp_path / "candidate.onnx"
    converted.write_bytes(b"not used because prerequisites are blocked")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not decoded because prerequisites are blocked")

    if command == "convert":
        argv = [
            command,
            str(MODEL),
            "--output",
            str(output),
            "--manifest-out",
            str(manifest),
        ]
    elif command == "validate":
        argv = [
            command,
            str(MODEL),
            str(converted),
            "--manifest-out",
            str(manifest),
        ]
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == f"wrote {manifest}\n"
    assert "blocked:" in captured.err
    assert not output.exists()

    record = json.loads(manifest.read_text())
    assert record["status"] == "blocked"
    assert record["gates"]["weight"] == "blocked"
    assert record["gates"]["quantization"] == "blocked"
    assert "weight_gate_blocked" in record["blockers"]
    assert "quantization_gate_blocked" in record["blockers"]
    assert record["outputs"] == []


def test_run_image_writes_exact_backend_result_and_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    image = tmp_path / "frame.png"
    image.write_bytes(b"exact image fixture")
    output = tmp_path / "result.json"
    manifest = tmp_path / "result.manifest.json"
    workspace = tmp_path / "runtime"
    observed: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, **kwargs):
            observed["runtime"] = kwargs

        def prepared(self):
            return object()

    def fake_run_exact_image(**kwargs):
        observed["inference"] = kwargs
        return {
            "schema": "verkeye.exact-image-inference.v1",
            "run_id": "cli-image-001",
            "detections": [],
        }

    monkeypatch.setattr("verkeye.cli.DockerAdesRuntime", FakeRuntime)
    monkeypatch.setattr("verkeye.cli.run_exact_image", fake_run_exact_image)

    result = main(
        [
            "run",
            str(MODEL),
            "--image",
            str(image),
            "--backend",
            "ades",
            "--workspace",
            str(workspace),
            "--run-id",
            "cli-image-001",
            "--json-out",
            str(output),
            "--manifest-out",
            str(manifest),
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == f"wrote {output}\nwrote {manifest}\n"
    assert json.loads(output.read_text())["schema"] == (
        "verkeye.exact-image-inference.v1"
    )
    record = json.loads(manifest.read_text())
    assert record["status"] == "passed"
    assert record["gates"]["exact_model_execution"] == "passed"
    assert record["gates"]["physical_cv22_parity"] == "pending"
    assert observed["inference"]["run_id"] == "cli-image-001"


def test_run_image_constructs_one_accelerated_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"fixture")
    for name in ("captures", "split4", "split5"):
        (tmp_path / name).mkdir()
    observed: dict[str, object] = {}

    class FakeSession:
        def __init__(self, **kwargs):
            observed["session"] = kwargs

    class FakeAcceleratedRuntime:
        def __init__(self, *, session):
            observed["backend_session"] = session
            observed["backend"] = self

    monkeypatch.setattr(
        "verkeye.cli.importlib.util.find_spec",
        lambda name: object() if name == "mlx" else None,
    )
    monkeypatch.setattr("verkeye.cli.Cb62MlxSession", FakeSession, raising=False)
    monkeypatch.setattr(
        "verkeye.cli.AcceleratedCb62Runtime",
        FakeAcceleratedRuntime,
        raising=False,
    )

    def fake_run_exact_image(**kwargs):
        observed["inference"] = kwargs
        return {
            "schema": "verkeye.exact-image-inference.v1",
            "run_id": "accelerated-fixture",
            "detections": [],
        }

    monkeypatch.setattr("verkeye.cli.run_exact_image", fake_run_exact_image)

    output = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    result = main(
        [
            "run",
            str(MODEL),
            "--image",
            str(image),
            "--backend",
            "accelerated",
            "--accelerated-capture-root",
            str(tmp_path / "captures"),
            "--accelerated-split4-root",
            str(tmp_path / "split4"),
            "--accelerated-split5-root",
            str(tmp_path / "split5"),
            "--json-out",
            str(output),
            "--manifest-out",
            str(manifest),
        ]
    )

    assert result == 0
    assert observed["session"] == {
        "capture_root": tmp_path / "captures",
        "split4_root": tmp_path / "split4",
        "split5_root": tmp_path / "split5",
    }
    assert observed["inference"]["backend"] is observed["backend"]


def test_run_image_auto_discovers_content_addressed_generated_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "owner-model.bin"
    model.write_bytes(b"owner model")
    image = tmp_path / "frame.png"
    image.write_bytes(b"fixture")
    generated_base = tmp_path / "generated"
    generated_root = generated_base / MODEL_SHA256
    generated_root.mkdir(parents=True)
    (generated_root / "manifest.json").write_text("{}")
    generated = object()
    observed: dict[str, object] = {}

    class FakeSession:
        def __init__(self, **kwargs: object) -> None:
            observed["session"] = kwargs

    class FakeAcceleratedRuntime:
        def __init__(self, *, session: object) -> None:
            observed["backend"] = self

    monkeypatch.setattr(
        "verkeye.cli.importlib.util.find_spec",
        lambda name: object() if name == "mlx" else None,
    )
    monkeypatch.setattr("verkeye.cli.Cb62MlxSession", FakeSession)
    monkeypatch.setattr(
        "verkeye.cli.AcceleratedCb62Runtime", FakeAcceleratedRuntime
    )
    monkeypatch.setattr(
        "verkeye.cli.load_generated_runtime",
        lambda root, expected_model_sha256: generated,
    )
    monkeypatch.setattr(
        "verkeye.cli.run_exact_image",
        lambda **kwargs: {
            "schema": "verkeye.exact-image-inference.v1",
            "run_id": "generated-runtime",
            "detections": [],
        },
    )

    result = main(
        [
            "run",
            str(model),
            "--image",
            str(image),
            "--backend",
            "accelerated",
            "--generated-runtime-base",
            str(generated_base),
            "--json-out",
            str(tmp_path / "result.json"),
            "--manifest-out",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 0
    assert observed["session"] == {"generated_runtime": generated}


def test_run_image_selects_openvino_gpu_when_mlx_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"fixture")
    for name in ("captures", "split4", "split5"):
        (tmp_path / name).mkdir()
    observed: dict[str, object] = {}

    class FakeCore:
        available_devices = ("CPU", "GPU")

    class FakeSession:
        def __init__(self, **kwargs):
            observed["session"] = kwargs

    class FakeAcceleratedRuntime:
        def __init__(self, *, session):
            observed["backend_session"] = session
            observed["backend"] = self

    monkeypatch.setattr(
        "verkeye.cli.importlib.util.find_spec",
        lambda name: object() if name == "openvino" else None,
    )
    fake_openvino = types.ModuleType("openvino")
    fake_openvino.Core = FakeCore
    monkeypatch.setitem(sys.modules, "openvino", fake_openvino)
    monkeypatch.setattr("verkeye.cli.OpenVinoCb62Session", FakeSession)
    monkeypatch.setattr(
        "verkeye.cli.AcceleratedCb62Runtime", FakeAcceleratedRuntime
    )
    monkeypatch.setattr(
        "verkeye.cli.run_exact_image",
        lambda **kwargs: {
            "schema": "verkeye.exact-image-inference.v1",
            "run_id": "openvino-fixture",
            "detections": [],
        },
    )

    result = main(
        [
            "run",
            str(MODEL),
            "--image",
            str(image),
            "--backend",
            "accelerated",
            "--accelerated-capture-root",
            str(tmp_path / "captures"),
            "--accelerated-split4-root",
            str(tmp_path / "split4"),
            "--accelerated-split5-root",
            str(tmp_path / "split5"),
            "--json-out",
            str(tmp_path / "result.json"),
            "--manifest-out",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 0
    assert observed["session"] == {
        "capture_root": tmp_path / "captures",
        "split4_root": tmp_path / "split4",
        "split5_root": tmp_path / "split5",
        "device": "GPU",
    }


@pytest.mark.parametrize(
    ("source_flag", "source_value", "source_kind"),
    [
        ("--frames", "frames", "frame-directory"),
        ("--video", "clip.mp4", "video"),
        ("--webcam", "2", "webcam"),
    ],
)
def test_run_stream_sources_use_the_exact_backend_and_bounded_frame_core(
    source_flag: str,
    source_value: str,
    source_kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "result.json"
    manifest = tmp_path / "result.manifest.json"
    workspace = tmp_path / "runtime"
    source_path = tmp_path / source_value
    if source_flag == "--frames":
        source_path.mkdir()
        (source_path / "frame1.png").write_bytes(b"frame")
    elif source_flag == "--video":
        source_path.write_bytes(b"video")
    observed: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, **kwargs):
            observed["runtime"] = kwargs

        def prepared(self):
            return object()

    def fake_frames(path):
        observed["frame_directory"] = path
        return iter(("frame-0", "frame-1", "frame-2"))

    def fake_capture(source, *, max_frames):
        observed["capture"] = (source, max_frames)
        return iter(("frame-0", "frame-1"))

    def fake_live_capture(source, *, queue_capacity, max_frames):
        observed["live_capture"] = (source, queue_capacity, max_frames)
        return iter(("frame-0", "frame-1"))

    def fake_run_exact_stream(**kwargs):
        observed["inference"] = kwargs
        assert list(kwargs["frames"])
        return {
            "schema": "verkeye.exact-stream-inference.v1",
            "run_id": "cli-stream-001",
            "source_kind": source_kind,
            "frame_count": 2,
            "execution": {"effective_backend_fps": 0.025},
        }

    monkeypatch.setattr("verkeye.cli.DockerAdesRuntime", FakeRuntime)
    monkeypatch.setattr("verkeye.cli.iter_frame_directory", fake_frames)
    monkeypatch.setattr("verkeye.cli.iter_capture", fake_capture)
    monkeypatch.setattr("verkeye.cli.LiveCaptureStream", fake_live_capture)
    monkeypatch.setattr("verkeye.cli.run_exact_stream", fake_run_exact_stream)

    argument_value = "2" if source_flag == "--webcam" else str(source_path)
    result = main(
        [
            "run",
            str(MODEL),
            source_flag,
            argument_value,
            "--max-frames",
            "2",
            "--workspace",
            str(workspace),
            "--run-id",
            "cli-stream-001",
            "--json-out",
            str(output),
            "--manifest-out",
            str(manifest),
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == f"wrote {output}\nwrote {manifest}\n"
    assert json.loads(output.read_text())["schema"] == (
        "verkeye.exact-stream-inference.v1"
    )
    record = json.loads(manifest.read_text())
    assert record["status"] == "passed"
    assert record["gates"]["exact_model_execution"] == "passed"
    assert record["gates"]["realtime_throughput"] == "not_claimed"
    assert observed["inference"]["run_id"] == "cli-stream-001"
    if source_flag == "--frames":
        assert observed["frame_directory"] == source_path
    elif source_flag == "--video":
        assert observed["capture"] == (source_path, 2)
    else:
        assert observed["live_capture"] == (2, 2, 2)


def test_exact_manifest_does_not_duplicate_runtime_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"fixture")
    output = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"

    class FakeRuntime:
        def __init__(self, **kwargs):
            pass

        def prepared(self):
            return object()

    monkeypatch.setattr("verkeye.cli.DockerAdesRuntime", FakeRuntime)
    monkeypatch.setattr(
        "verkeye.cli.run_exact_image",
        lambda **kwargs: {
            "schema": "verkeye.exact-image-inference.v1",
            "run_id": "manifest-inputs",
            "detections": [],
        },
    )

    assert main(
        [
            "run",
            str(MODEL),
            "--image",
            str(image),
            "--json-out",
            str(output),
            "--manifest-out",
            str(manifest),
        ]
    ) == 0

    inputs = json.loads(manifest.read_text())["inputs"]
    paths = [item["path"] for item in inputs]
    assert len(paths) == len(set(paths))


def test_webcam_requires_an_explicit_frame_bound(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    result = main(
        [
            "run",
            str(MODEL),
            "--webcam",
            "0",
            "--json-out",
            str(tmp_path / "result.json"),
            "--manifest-out",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 1
    captured = capsys.readouterr()
    assert "--webcam requires --max-frames" in captured.err


def test_live_parser_selects_camera_and_viewer_defaults() -> None:
    args = _parser().parse_args(
        [
            "live",
            str(MODEL),
            "--webcam",
            "2",
        ]
    )

    assert args.webcam == 2
    assert args.video is None
    assert args.backend == "auto"
    assert args.queue_size == 2
    assert args.max_frames is None
    assert args.record is None
    assert args.record_fps == 30.0
    assert args.record_codec == "mp4v"
    assert args.summary_out == Path("verkeye-live-summary.json")


def test_live_parser_requires_exactly_one_media_source() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["live", str(MODEL)])
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "live",
                str(MODEL),
                "--webcam",
                "0",
                "--video",
                "clip.mp4",
            ]
        )


def test_live_command_reuses_exact_backend_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "viewer-summary.json"
    record_path = tmp_path / "annotated.mp4"
    observed: dict[str, object] = {}
    spec = object()
    backend = object()

    monkeypatch.setattr("verkeye.cli._exact_backend", lambda args: (spec, backend))

    class FakeExactFrameSession:
        def __init__(self, **kwargs):
            observed["exact"] = kwargs

    class FakeSource:
        def __init__(self, source, **kwargs):
            observed["source"] = (source, kwargs)

    class FakeDisplay:
        pass

    writer = object()

    def fake_open_writer(path, **kwargs):
        observed["writer"] = (path, kwargs)
        return writer

    class FakeViewer:
        def __init__(self, **kwargs):
            observed["viewer"] = kwargs

        def run(self):
            return {
                "schema": "verkeye.live-viewer-session.v1",
                "run_id": "live-cli-fixture",
            }

    monkeypatch.setattr("verkeye.cli.ExactFrameSession", FakeExactFrameSession)
    monkeypatch.setattr("verkeye.cli.LiveCaptureStream", FakeSource)
    monkeypatch.setattr("verkeye.cli.OpenCvDisplay", FakeDisplay)
    monkeypatch.setattr("verkeye.cli.open_video_writer", fake_open_writer)
    monkeypatch.setattr("verkeye.cli.LiveViewerSession", FakeViewer)

    result = main(
        [
            "live",
            str(MODEL),
            "--webcam",
            "3",
            "--record",
            str(record_path),
            "--record-fps",
            "24",
            "--max-frames",
            "4",
            "--summary-out",
            str(summary_path),
        ]
    )

    assert result == 0
    assert json.loads(summary_path.read_text())["run_id"] == "live-cli-fixture"
    assert observed["exact"] == {
        "model": MODEL,
        "backend": backend,
        "runtime_spec": spec,
        "pipeline_evidence": _default_runtime_asset(
            "evidence/compatibility/cvproc-yolov6-pipeline.json"
        ),
        "operating_profile": "small-object",
    }
    assert observed["source"] == (3, {"queue_capacity": 2, "max_frames": None})
    assert observed["writer"] == (
        record_path,
        {"fps": 24.0, "codec": "mp4v"},
    )
    viewer = observed["viewer"]
    assert viewer["writer"] is writer
    assert viewer["recording"]["path"] == str(record_path.resolve())
    assert viewer["max_frames"] == 4
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == f"wrote {summary_path}\n"


def test_cli_reports_missing_input_without_emitting_malformed_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    output = tmp_path / "inspect.json"
    manifest = tmp_path / "inspect.manifest.json"
    result = main(
        [
            "inspect",
            str(tmp_path / "missing.bin"),
            "--json-out",
            str(output),
            "--manifest-out",
            str(manifest),
        ]
    )

    assert result == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert not output.exists()
    assert not manifest.exists()
