import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PUBLIC_REFERENCES = (
    "/Users/" + "nigel",
    "/mnt/" + "SECOND",
    "gainpal" + "fam.com",
)
RECOVERED_ARTIFACTS = (
    "fixtures/models/yolov6n_hor.bin",
    "fixtures/models/yolov6n_ver.bin",
    "fixtures/vendor/cv22",
)
PUBLIC_REPOSITORY_URL = "https://github.com/GainSec/VerkEye"
REQUIRED_PUBLIC_EVIDENCE = {
    "evidence/README.md",
    "evidence/compatibility/cvproc-yolov6-pipeline.json",
    "evidence/demos/demo-1-live-smoke.json",
    "evidence/demos/demo-1-screenshot.png",
    "evidence/demos/demo-2-annotated.png",
    "evidence/demos/demo-2-fixture-acceptance.json",
    "evidence/demos/demo-2-screenshot.png",
    "evidence/demos/demo-2-session.json",
    "evidence/final-verification.txt",
    "evidence/equivalence/fresh-clone-verification.txt",
    "evidence/equivalence/kali-openvino-terminal-parity.json",
    "evidence/equivalence/macos-mlx-terminal-parity.json",
    "evidence/equivalence/production-detection-parity.json",
    "evidence/performance/kali-openvino-equivalent-final.json",
    "evidence/performance/macos-mlx-equivalent-final.json",
    "evidence/public-webcams/data-gov-sg-20261003/results/"
    "camera-2704-small-object.png",
    "evidence/public-webcams/small-object-validation-20261003.json",
}


def test_owner_generated_runtime_is_gitignored_and_not_manifested():
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", ".runtime/generated/example/manifest.json"],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
    assert not (ROOT / "config/cb62-generated-runtime-assets.json").exists()


def test_public_release_excludes_all_recovered_device_artifacts():
    for relative_path in RECOVERED_ARTIFACTS:
        assert not (ROOT / relative_path).exists(), relative_path

    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert "/fixtures/models/yolov6n_hor.bin" in ignore
    assert "/fixtures/models/yolov6n_ver.bin" in ignore
    assert "/fixtures/vendor/cv22/" in ignore

    package = (ROOT / "pyproject.toml").read_text()
    assert '"share/verkeye/models"' not in package


def test_tracked_public_text_has_no_private_workstation_or_host_references():
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    violations: list[str] = []
    for raw_path in tracked:
        if not raw_path:
            continue
        relative_path = raw_path.decode()
        path = ROOT / relative_path
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if b"\0" in payload:
            continue
        text = payload.decode("utf-8", errors="ignore")
        for forbidden in FORBIDDEN_PUBLIC_REFERENCES:
            if forbidden in text:
                violations.append(f"{relative_path}: {forbidden}")
    assert not violations, "\n".join(violations)


def test_public_tree_excludes_internal_workbench_transport_records():
    assert not (ROOT / "evidence/workbench").exists()
    assert not (ROOT / "workbench/publication.json").exists()


def test_public_readme_has_no_staging_placeholders_or_stale_test_count():
    readme = (ROOT / "README.md").read_text()
    assert "<public-VerkEye-repository-url>" not in readme
    assert "472 passed, 4 skipped" not in readme


def test_public_readme_documents_owner_model_and_local_runtime_generation():
    readme = (ROOT / "README.md").read_text()
    for statement in (
        "This repository intentionally excludes recovered camera artifacts",
        "Extract `yolov6n_hor.bin` from a CB62 you own",
        "verkeye generate-runtime",
        "Generated runtime parameters are not distributed",
    ):
        assert statement in readme


def test_workbench_demo_outputs_are_included_with_published_hashes():
    expected = {
        "demo-1-live-smoke.json": "cd4b63760d1872b14524109089cf6aaf14105de24a2c3bc33ec1a822bcf77335",
        "demo-1-screenshot.png": "b07accd28594a556edbd4aa0dbe5f39eeccfe01a88ced7420a42dcfda69c4f8b",
        "demo-2-annotated.png": "8569485853d82c7f3b74817c06f91ae6e10eb5f054df0b8533838ef630d7d3b5",
        "demo-2-fixture-acceptance.json": "48d42f5bc132461c1b5a5b2fecaa3f415f44d71ed07c33c16549f32fc17e9ba3",
        "demo-2-screenshot.png": "7202437ab2c00aab8ee0ee55d99544af2c77c8a56c2496c431146e21caf584d6",
        "demo-2-session.json": "ad840e57a21334ab84db4c09d6b1dd6db84dfa495369678c707358c382efd4b3",
    }
    for filename, digest in expected.items():
        payload = (ROOT / "evidence/demos" / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest


def test_public_readme_embeds_annotated_demo_outputs():
    readme = (ROOT / "README.md").read_text()
    for image in (
        "evidence/demos/demo-2-annotated.png",
        "evidence/demos/demo-2-screenshot.png",
    ):
        assert f"]({image})" in readme
        assert (ROOT / image).is_file()
    assert "Green boxes are accepted production-profile detections" in readme


def test_public_release_manifest_records_source_and_boundary():
    release = (ROOT / "PUBLIC_RELEASE.md").read_text()
    assert "1233882ef2ac3fe240fc18fab3ece7cab865caa2" in release
    assert "fixtures/models/yolov6n_hor.bin" in release
    assert "5,561,124 bytes" in release
    assert "eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf" in release


def test_public_evidence_tree_is_the_minimal_runtime_and_release_proof_set():
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "evidence").rglob("*")
        if path.is_file()
    }
    assert actual == REQUIRED_PUBLIC_EVIDENCE


def test_public_documents_use_the_final_github_url():
    for relative_path in ("README.md", "PUBLIC_RELEASE.md"):
        text = (ROOT / relative_path).read_text()
        assert PUBLIC_REPOSITORY_URL in text, relative_path
