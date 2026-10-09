import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKBENCH = ROOT / "workbench"
HTML_FILES = (
    "overview.html",
    "model-anatomy.html",
    "conversion-status.html",
    "research-journal.html",
    "local-runner-guide.html",
    "live-viewer.html",
    "public-webcam-validation.html",
    "literal-equivalence.html",
    "one-command-demos.html",
)
MODEL_SHA256 = "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf"
MACOS_CORPUS = "<owner-corpus-root>/evidence/approot/cb62-specimen2-approot-ml-corpus-20260927.tar.zst"
KALI_MODEL = "<owner-storage>/bull-cam-2-non512/offline-approot/b/var/cache/cvproc/yolov6n_hor.bin"


def test_all_approved_workbench_documents_exist_and_use_theme_tokens():
    for filename in HTML_FILES:
        path = WORKBENCH / filename
        assert path.exists(), f"missing {path}"
        html = path.read_text()
        assert html.startswith("<!DOCTYPE html>")
        assert "var(--wb-bg" in html
        assert "var(--wb-accent" in html
        assert "viewport" in html


def test_document_set_contains_exact_portable_evidence_references():
    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    assert MODEL_SHA256 in corpus
    assert MACOS_CORPUS in corpus
    assert KALI_MODEL in corpus
    assert "prim_split_0" in corpus
    assert "prim_split_9" in corpus
    assert not re.search(r"\bredacted\b|\[redaction\]|<redacted>", corpus, re.I)


def test_overview_visualizes_the_conversion_pipeline():
    html = (WORKBENCH / "overview.html").read_text()
    for label in (
        "Recovered Verkada .bin",
        "Digest-pinned Ambarella image",
        "Verified split executor",
        "Exact detector ABI",
        "VerkEye media runtime",
    ):
        assert label in html
    assert 'class="flow"' in html


def test_evidence_details_are_collapsed_by_default():
    for filename in ("model-anatomy.html", "research-journal.html"):
        html = (WORKBENCH / filename).read_text()
        details = re.findall(r"<details(?:\s+[^>]*)?>", html)
        assert details, f"expected collapsed evidence in {filename}"
        assert all(" open" not in tag for tag in details)


def test_task_board_tracks_each_feasibility_gate():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    assert board["title"] == "VerkEye implementation and feasibility gates"
    steps = {step["id"]: step for step in board["steps"]}
    for gate in (
        "container-gate",
        "tensor-gate",
        "weight-gate",
        "quantization-gate",
        "graph-gate",
        "inference-gate",
    ):
        assert gate in steps
        assert steps[gate]["status"] in {"todo", "in-progress", "done", "blocked"}


def test_container_and_tensor_gate_results_are_published_with_raw_evidence():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["container-gate"]["status"] == "done"
    assert steps["tensor-gate"]["status"] == "done"

    corpus = "\n".join(
        (WORKBENCH / name).read_text()
        for name in (
            "model-anatomy.html",
            "conversion-status.html",
            "research-journal.html",
        )
    )
    assert "39 tensor descriptors" in corpus
    assert "16 split connections" in corpus
    assert "0x480-byte" in corpus
    assert "1 × 3 × 608 × 1088" in corpus
    assert (
        "<repo-root>/evidence/"
        "yolov6n_hor.tensor-map.json"
    ) in corpus


def test_weight_and_quantization_blockers_are_published_without_false_onnx_claims():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["weight-gate"]["status"] == "blocked"
    assert steps["quantization-gate"]["status"] == "blocked"
    assert steps["graph-gate"]["status"] == "blocked"
    assert steps["inference-gate"]["status"] == "blocked"

    corpus = "\n".join(
        (WORKBENCH / name).read_text()
        for name in (
            "overview.html",
            "model-anatomy.html",
            "conversion-status.html",
            "research-journal.html",
            "local-runner-guide.html",
        )
    )
    for evidence in (
        "5,514,624",
        "opaque_compiled_package",
        "3,603,701",
        "semantic_parameter_bytes=0",
        "proprietary_nnctrl_decoder_unavailable",
        "<repo-root>/evidence/"
        "yolov6n_hor.weight-map.json",
    ):
        assert evidence in corpus
    assert "Exact ONNX export is stopped by design" in corpus
    assert "exact recovered model executes camera-free" in corpus.lower()
    assert "Direct ONNX" in corpus


def test_lossless_ir_completion_and_operator_boundary_are_published():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["lossless-ir"]["status"] == "done"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "VerkEye IR v1",
        "10 opaque split nodes",
        "zero claimed constants",
        "cv22.opaque_compiled_split",
        "verkeye ir MODEL.bin --json-out MODEL.ir.json",
        "<repo-root>/evidence/yolov6n_hor.ir.json",
        "b0932967cd1f1347874d27240f2a2b567eea65ef945b033bc762a505b53ff679",
    ):
        assert value in corpus


def test_exact_descriptor_abi_is_published_without_overclaiming_quantization():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["descriptor-abi"]["status"] == "done"
    assert steps["quantization-gate"]["status"] == "blocked"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "gen_net_parent_port_size",
        "0x8398",
        "647b38c0ef77558bc58bc2b32e1003bd8c4c781eaf25e195268297edc44f4c3d",
        "1ef8cfadb7218afb945067348f8bfc40b03712de1f47fbc1cfba0f80ef35cc46",
        "recovered_tensor_storage_dtypes=39",
        "recovered_tensor_exponent_offsets=39",
        "<repo-root>/evidence/compatibility/descriptor-abi.json",
        "181 passed in 6.85s",
        "181 passed in 14.06s",
    ):
        assert value in corpus

    assert "numeric-value formula" in corpus
    assert "semantic floating encoding" in corpus


def test_media_pipeline_publishes_exact_image_and_stream_modes():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["media-pipeline"]["status"] == "done"
    assert steps["local-runner"]["status"] == "done"
    assert steps["cb62-backend"]["status"] == "done"
    assert steps["ades-offline-image"]["status"] == "done"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "naturally ordered frame directories",
        "class-aware NMS",
        "no implicit preprocessing defaults",
        "13,566 × 8 float32",
        "35.827 seconds",
        "35.220 seconds",
        "0.0284 FPS",
        "22 raw tensors",
    ):
        assert value in corpus
    assert "The recovered CB62 output layout remains unknown" not in corpus


def test_reference_backend_contract_is_published_without_false_cb62_execution():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["reference-backend-contract"]["status"] == "done"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "Inspectable backend contract",
        "cv22.opaque_compiled_split",
        "deterministic SHA-256 records for every intermediate",
        "225 complete macOS checks",
        "<repo-root>/evidence/compatibility/"
        "reference-backend.txt",
    ):
        assert value in corpus
    assert "production opaque graph executes in the reference backend" not in corpus


def test_exact_cb62_postprocessor_checkpoint_is_published_fail_closed():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["cb62-postprocessor-core"]["status"] == "done"
    assert steps["cb62-operational-config"]["status"] == "done"
    assert steps["cb62-preprocessing"]["status"] == "done"
    assert steps["local-runner"]["status"] == "done"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "434,112 bytes",
        "0x5017E0",
        "ffd295e8382bf29877f4c54be1d9a37948531fc2d16c22c6528b30dbe598b405",
        "<repo-root>/evidence/compatibility/"
        "cvproc-yolov6-pipeline.json",
        "full-range BT.601",
        "uint8[1,3,608,1088]",
        "0 person",
        "1 vehicle",
        "2 animal",
        "v2026.07.30.926397-bruce-4k",
        "022d8371a822b702cd7c0cbc81b7280a84b2f724fb52ab8352fd92a38f346908",
        "32609cc9ea84f9f2887c951a33b432177b6592e594ba458eae088c3b52900255",
        "person-threshold = 0.875",
        "vehicle-threshold = 0.921875",
        "dedup-iou-threshold = 0.8",
        "class IDs 0→1→2",
    ):
        assert value in corpus

    assert "four operational inputs remain unresolved" not in corpus
    assert "operational-threshold-overrides-unresolved" not in corpus
    assert "secondary-dedup-parameters-unresolved" not in corpus
    assert "Exact image and video — operational" in corpus
    assert "bounded webcam" in corpus.lower()


def test_fail_closed_cli_and_manifests_are_published_as_completed():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["evidence-cli"]["status"] == "done"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "verkeye map MODEL.bin --json-out",
        "--manifest-out",
        "verkeye.execution-manifest.v1",
        "atomic",
        "structured blocker",
        "93 project checks",
        "commit <code>15c4b4f</code>",
    ):
        assert value in corpus
    assert "CLI not exposed" not in corpus
    assert "not implemented" not in corpus


def test_readme_and_final_verification_explain_supported_scope():
    readme = ROOT / "README.md"
    verification = ROOT / "evidence" / "final-verification.txt"
    assert readme.exists()
    assert verification.exists()

    content = readme.read_text()
    for value in (
        "VerkEye",
        MODEL_SHA256,
            "pip install -e '.[dev,runtime,macos]'",
            "pip install -e '.[dev,viewer,macos]'",
            "pip install -e '.[dev,runtime,linux]'",
            "pip install -e '.[dev,viewer,linux]'",
        "verkeye inspect",
        "verkeye map",
        "verkeye ir",
        "verkeye convert",
        "verkeye validate",
        "verkeye run",
        "exit with code 2",
        "Exact camera-free ADES runtime",
        "--video clip.mkv --max-frames 10",
        "--webcam 0 --max-frames 10 --queue-size 2",
    ):
        assert value in content

    log = verification.read_text()
    assert "346 passed" in log
    assert "webcam_queue_policy=drop-oldest" in log
    assert "kali_regression=PASS" in log
    assert "exact_still_image=PASS" in log
    assert "exact_file_video=PASS" in log
    assert "installed_wheel_exact_inference=PASS" in log
    assert "wheel_install_with_runtime_extra=PASS" in log


def test_exact_accelerated_runtime_is_published_with_current_performance() -> None:
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["accelerated-runtime"]["status"] == "done"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    readme = (ROOT / "README.md").read_text()
    for value in (
        "MLX",
        "OpenVINO",
        "32.236698 FPS",
        "3.051105 FPS",
        "--backend accelerated",
        "29-case",
        "174 terminal tensors",
        "zero differing bytes",
        "byte-identical",
    ):
        assert value in corpus
        assert value in readme

    assert "Current host-emulator throughput is 0.0284 FPS" not in corpus
    assert "Real-time throughput | Not achieved" not in readme
    assert "small score-head drift" not in corpus
    assert "small score-head drift" not in readme
    assert "four independent ADES oracle inputs" not in corpus
    assert "four independent ADES oracle inputs" not in readme


def test_literal_equivalence_report_covers_the_full_proof_boundary() -> None:
    html = (WORKBENCH / "literal-equivalence.html").read_text()
    readme = (ROOT / "README.md").read_text()
    corpus = html + "\n" + readme

    for value in (
        MODEL_SHA256,
        "identical uint8 CHW bytes entering the recovered CB62 model",
        "Ambarella ADES",
        "29-case",
        "174 terminal tensors",
        "zero differing bytes",
        "13,566 × 8",
        "production",
        "small-object",
        "32.236698 FPS",
        "3.051105 FPS",
        "scripts/verify_accelerated_equivalence.py",
        "scripts/benchmark_accelerated_equivalence.py",
        "macos-mlx-terminal-parity.json",
        "kali-openvino-terminal-parity.json",
        "production-detection-parity.json",
    ):
        assert value in corpus

    assert 'class="flow"' in html
    details = re.findall(r"<details(?:\s+[^>]*)?>", html)
    assert details
    assert all(" open" not in tag for tag in details)


def test_literal_equivalence_machine_evidence_is_self_consistent() -> None:
    mlx = json.loads(
        (ROOT / "evidence/equivalence/macos-mlx-terminal-parity.json").read_text()
    )
    openvino = json.loads(
        (ROOT / "evidence/equivalence/kali-openvino-terminal-parity.json").read_text()
    )
    detection = json.loads(
        (ROOT / "evidence/equivalence/production-detection-parity.json").read_text()
    )
    mac_perf = json.loads(
        (ROOT / "evidence/performance/macos-mlx-equivalent-final.json").read_text()
    )
    kali_perf = json.loads(
        (ROOT / "evidence/performance/kali-openvino-equivalent-final.json").read_text()
    )

    for report in (mlx, openvino):
        assert report["status"] == "passed"
        assert report["case_count"] == 29
        assert report["terminal_tensor_count"] == 174
        assert report["terminal_tensor_differing_bytes"] == 0
        assert report["prediction_mismatch_count"] == 0
        assert report["production_detection_mismatch_count"] == 0

    assert detection["status"] == "passed"
    assert detection["oracle_catalog"]["case_count"] == 29
    assert detection["oracle_catalog"]["prediction_matrix_mismatch_count"] == 0
    assert detection["oracle_catalog"]["decoded_detection_mismatch_count"] == 0
    assert detection["nonempty_contract_witness"]["accepted_detection_count"] == 2

    assert mac_perf["status"] == "passed"
    assert mac_perf["correctness_gate"]["before_benchmark"] == "passed"
    assert mac_perf["correctness_gate"]["after_benchmark"] == "passed"
    assert kali_perf["status"] == "below_target"
    assert kali_perf["correctness_gate"]["before_benchmark"] == "passed"
    assert kali_perf["correctness_gate"]["after_benchmark"] == "passed"


def test_fresh_clone_asset_and_media_contract_is_documented() -> None:
    readme = (ROOT / "README.md").read_text()
    owner_generation = (ROOT / "docs/OWNER_RUNTIME_GENERATION.md").read_text()
    corpus = readme + "\n" + owner_generation

    for value in (
        "verkeye generate-runtime",
        ".runtime/generated/",
        "55 valid fast-convolution captures",
        "Generated runtime parameters are not distributed",
        "verkeye run",
        "--image",
        "--webcam",
        "--generated-runtime-base",
    ):
        assert value in corpus


def test_live_viewer_is_published_with_commands_controls_and_raw_evidence() -> None:
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["live-viewer"]["status"] == "done"

    viewer = (WORKBENCH / "live-viewer.html").read_text()
    readme = (ROOT / "README.md").read_text()
    corpus = viewer + "\n" + readme
    for value in (
        "verkeye live",
        "--webcam 0",
        "--video",
        "--record",
        "Space / P",
        "Q / Esc",
        "centered-letterbox",
        "1088 × 608",
        "person",
        "vehicle",
        "animal",
        "Capture FPS",
        "Inference FPS",
        "Display FPS",
        "31.88 FPS",
        "30.26 FPS",
        "300 captured",
        "zero drops",
        "472 passed, 4 skipped",
        "exact recovered CB62 model",
        "<repo-root>/evidence/viewer/"
        "macos-prerecorded-summary.json",
        "<repo-root>/evidence/viewer/"
        "macos-prerecorded-frame.png",
        "<repo-root>/evidence/viewer/"
        "macos-native-window-summary.json",
    ):
        assert value in corpus

    assert 'class="flow"' in viewer
    assert "proxy model" in viewer.lower()
    assert "No physical webcam was enumerated" in viewer
    assert "pip install -e '.[dev,viewer,macos]'" in readme


def test_live_viewer_workbench_evidence_is_collapsed_by_default() -> None:
    html = (WORKBENCH / "live-viewer.html").read_text()
    details = re.findall(r"<details(?:\s+[^>]*)?>", html)
    assert details
    assert all(" open" not in tag for tag in details)


def test_small_object_profile_and_public_detection_are_published() -> None:
    viewer = (WORKBENCH / "live-viewer.html").read_text()
    validation = (WORKBENCH / "public-webcam-validation.html").read_text()
    corpus = viewer + validation + (WORKBENCH / "research-journal.html").read_text()

    for value in (
        "--detection-profile small-object",
        "minimum normalized area",
        "0.001",
        "0.00025",
        "person 0.101",
        "camera 2704",
        "1 accepted detection",
        "small-object-results.json",
    ):
        assert value in corpus

    assert "live viewer defaults to" in viewer.lower()
    assert "production profile remains" in corpus.lower()
    assert (
        ROOT
        / "evidence/public-webcams/data-gov-sg-20261003/results/"
        "camera-2704-small-object.png"
    ).is_file()
    assert (
        ROOT / "evidence/public-webcams/small-object-validation-20261003.json"
    ).is_file()


def test_recovered_run_dags_driver_contract_is_published_without_overclaiming():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["cavalry-driver-contract"]["status"] == "done"
    assert steps["visorc-boot-contract"]["status"] == "done"
    assert steps["accelerator-execution"]["status"] == "in-progress"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "CAVALRY_RUN_DAGS",
        "0xc0084303",
        "2,860-byte",
        "128 primary",
        "64 secondary",
        "0x5F1FC",
        "0x1400",
        "0x80000001",
        "0x80000002",
        "stop response",
        "VISORC boot",
        "0xED000000",
        "0x80228",
        "0x28000",
        "62 instruction-byte proofs",
        "a408c6df40ac75278abf6af453c2eaeeb93e87e10a0912a6cc78863601f4b796",
        "39811bf17113e78c7babe9c0a6b8bad53ca80648847c364da5fccb5dd5f0de24",
        "7e49a1378c7ebc825786c4ec212d07d9ddd60b57870f248cee1367e8d3db5c5a",
        "189 passed in 6.93s",
        "189 passed in 14.29s",
    ):
        assert value in corpus

    assert "accelerator computation remains unsupported" in corpus
    assert "inference_produced=false" in corpus


def test_cv22_openrisc_discovery_checkpoint_is_published_fail_closed():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["openrisc-boundary-discovery"]["status"] == "done"
    assert steps["accelerator-execution"]["status"] == "in-progress"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "23 trace-proven boundaries",
        "0x07 and 0x10 are unallocated",
        "0x1c = l.cust1",
        "0x1f = l.cust4",
        "timeout_without_illegal_instruction_boundary",
        "failed_by_construction",
        "openrisc-discovery-run-001.json",
        "a6fdcd3",
        "263 passed in 15.14s",
    ):
        assert value in corpus

    assert "opcode 0x07 as l.cust1" not in corpus
    assert "reaches l.cust1 with captured operands" not in corpus


def test_cv22_no_delay_correction_is_published_and_supersedes_stale_contexts():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    discovery = steps["openrisc-boundary-discovery"]
    execution = steps["accelerator-execution"]

    assert "0fc223db6330f5e54b6c90dfd1585a42ddf3f2eaff8803c2e0d706965d29b6c5" in discovery["verification"]
    assert "21 register snapshots" in discovery["verification"]
    assert "superseded context snapshots" in execution["verification"]

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "openrisc-control-flow-correction.json",
        "openrisc-discovery-run-002.json",
        "openrisc-context-run-002.json",
        "0x004120cc",
        "0x004120d0",
        "21 context snapshots",
        "failed_by_construction",
    ):
        assert value in corpus


def test_cv22_custom_instruction_surface_and_constraint_ledger_are_published():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    surface = steps["openrisc-custom-surface"]
    execution = steps["accelerator-execution"]

    assert surface["status"] == "done"
    assert "26 proven occurrences" in surface["verification"]
    assert "23 dynamic" in surface["verification"]
    assert "3 static-only" in surface["verification"]
    assert "427 conditional" in surface["verification"]
    assert "367 excluded" in surface["verification"]
    assert "bounded model covers 0" in execution["verification"]

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "openrisc-custom-surface.json",
        "openrisc-constraint-ledger.json",
        "91834bfb2e737b1ff987ae6c5cedee72a4fbfbb547688fba36b9818a94e5f48a",
        "ed6fa5bc989fc5416e91ed87458c27cce0c854f67377cf67e6a7ce16dd9dccf8",
        "26 proven occurrences",
        "23 dynamic contexts",
        "3 static-only",
        "427 conditional",
        "367 excluded",
        "1 exact context",
        "22 tainted contexts",
        "zero of the 26",
        "CV22_CUSTOM_INSTRUCTION_SEMANTICS_UNRESOLVED",
    ):
        assert value in corpus

    assert "605 instructions execute" not in corpus


def test_public_cavalry_and_run_dags_semantic_correlations_are_published():
    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["public-cavalry-abi"]["status"] == "done"
    assert steps["run-dags-semantic-abi"]["status"] == "done"

    corpus = "\n".join((WORKBENCH / name).read_text() for name in HTML_FILES)
    for value in (
        "35 of 38",
        "71b1b786689f690248524b5139838b0a778047c7",
        "aa3856db0028691b688b122a11279d87efd25f65",
        "b6b4197498d81a8c5dfa393a0516ebb179c9be889e72d58245c7038ca74bca6e",
        "18e22466b1a56842c4321e62141022151faf04724d3313a7480a7aa43182c927",
        "dvi_dram_addr",
        "dvi_img_vaddr",
        "dvi_img_size",
        "dvi_dag_vaddr",
        "port_boffset_in_dag",
        "port_daddr_increment",
        "poke_vaddr",
        "10 descriptors",
        "39 port records",
        "zero poke records",
        "four prefix words",
        "320 complete macOS checks",
    ):
        assert value in corpus

    assert "public headers are the exact historical CB62 SDK" not in corpus
    assert "firmware-side accelerator semantics are resolved" not in corpus


def test_one_command_demos_are_documented_with_pinned_evidence() -> None:
    page = (WORKBENCH / "one-command-demos.html").read_text()
    readme = (ROOT / "README.md").read_text()
    corpus = page + "\n" + readme

    for value in (
        "verkeye --demo 1",
        "verkeye --demo 2",
        "small-object",
        "production",
        "Singapore LTA",
        "CC0 1.0",
        "459f3696a29606084bbb18e3e2acb2f435f72c31f74f5637c480db0267109741",
        "365,478 bytes",
        "person 0.101",
        "No user media is required",
        "demo-1-screenshot.png",
        "demo-2-screenshot.png",
        "demo-1-live-smoke.json",
        "demo-2-session.json",
        "Q / Esc",
        "Space / P",
    ):
        assert value in corpus

    assert 'class="flow"' in page
    assert "/Users/" not in page
    assert "/mnt/" not in page

    board = json.loads((WORKBENCH / "task-board.json").read_text())
    steps = {step["id"]: step for step in board["steps"]}
    assert steps["one-command-demos"]["status"] == "done"
