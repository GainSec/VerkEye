from __future__ import annotations

from pathlib import Path

from verkeye.demo import (
    DemoPreset,
    LoopingFixtureSource,
    demo_preset,
    resolve_demo_runtime_paths,
    run_demo,
)
from verkeye.runtime.demo_sources import SingaporeTrafficStream


def test_demo_presets_are_fixed_and_operator_legible() -> None:
    first = demo_preset(1)
    second = demo_preset(2)

    assert first == DemoPreset(
        number=1,
        profile="small-object",
        source_label="Singapore LTA traffic stills",
        banner="DEMO 1 | SINGAPORE LTA | SMALL-OBJECT PROFILE",
        window_title="VerkEye Demo 1 — Singapore traffic / small-object",
    )
    assert second == DemoPreset(
        number=2,
        profile="production",
        source_label="Bundled CC0 street scene",
        banner="DEMO 2 | BUNDLED OFFLINE SCENE | PRODUCTION PROFILE",
        window_title="VerkEye Demo 2 — bundled scene / production",
    )


def test_demo_runtime_paths_resolve_all_exact_assets() -> None:
    paths = resolve_demo_runtime_paths()

    assert paths.model.name == "yolov6n_hor.bin"
    assert paths.runtime_spec.name == "cb62-ades-runtime.json"
    assert paths.pipeline_evidence.name == "cvproc-yolov6-pipeline.json"
    assert paths.generated_runtime_root.is_dir()


def test_demo_source_selection_uses_network_then_bundled_fixture() -> None:
    first = demo_preset(1).make_source()
    second = demo_preset(2).make_source()
    try:
        assert isinstance(first, SingaporeTrafficStream)
        assert isinstance(second, LoopingFixtureSource)
        assert second.stats_snapshot()["license"] == "CC0 1.0"
        assert second.stats_snapshot()["sha256"] == (
            "459f3696a29606084bbb18e3e2acb2f435f72c31f74f5637c480db0267109741"
        )
    finally:
        first.close()
        second.close()


def test_run_demo_routes_profile_source_banner_and_summary(tmp_path: Path) -> None:
    events: dict[str, object] = {}

    class FakeViewer:
        def __init__(self, **kwargs: object) -> None:
            events.update(kwargs)

        def run(self) -> dict[str, object]:
            return {"exit_reason": "key-quit"}

    def session_factory(preset: DemoPreset, _paths: object) -> object:
        events["profile"] = preset.profile
        return object()

    def source_factory(preset: DemoPreset) -> object:
        events["source_preset"] = preset.number
        return object()

    summary = tmp_path / "demo-1.json"
    result = run_demo(
        1,
        summary_path=summary,
        source_factory=source_factory,
        session_factory=session_factory,
        display_factory=object,
        viewer_factory=FakeViewer,
    )

    assert result == 0
    assert events["profile"] == "small-object"
    assert events["source_preset"] == 1
    assert events["source_label"] == "Singapore LTA traffic stills"
    assert events["banner"] == "DEMO 1 | SINGAPORE LTA | SMALL-OBJECT PROFILE"
    assert events["window_title"] == (
        "VerkEye Demo 1 — Singapore traffic / small-object"
    )
    assert events["summary_path"] == summary

