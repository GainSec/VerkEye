# VerkEye public release manifest

This repository is a fresh-history public snapshot of internal VerkEye commit
`1233882ef2ac3fe240fc18fab3ece7cab865caa2`.

## Excluded owner artifacts

The snapshot excludes all recovered camera artifacts, including:

```text
fixtures/models/yolov6n_hor.bin
fixtures/models/yolov6n_ver.bin
fixtures/vendor/cv22/
```

Owners extract these artifacts from their own CB62 and place them in the
ignored paths above when reproducing forensic tests. The normal VerkEye runtime
requires only the horizontal model; the CV22 vendor binaries, libraries,
driver, firmware, bytecode, and production configuration are not loaded by
image, video, demo, or live-viewer inference.

The required artifact is exactly 5,561,124 bytes with SHA-256:

```text
eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf
```

The paths are ignored by Git, and public package metadata does not embed them.

## Included generated runtime

VerkEye's generated compatibility runtime is included directly in this
repository:

| Directory | Files | Bytes | Canonical tree SHA-256 |
|---|---:|---:|---|
| `.runtime/ades-full-kernels` | 7,519 | 100,167,528 | `92840f3b25d4b8a3c3664e72e0889898e1df4580c6b500d262d6713542253418` |
| `.runtime/ades-split4` | 46 | 8,839,183 | `e3705740eff64fac2e58a915cd61fc2059af3e85a977e0529937c73c16ceee73` |
| `.runtime/ades-split5` | 32 | 11,309,532 | `cf30293fcb46908ce908a6476fd3dfb0f7dc82d2a764968d819423f67c27cff6` |

The machine-readable record is
`config/cb62-generated-runtime-assets.json`.

## Included Workbench demo output

These are the run outputs published on the VerkEye Workbench:

| File | SHA-256 |
|---|---|
| `evidence/demos/demo-1-live-smoke.json` | `cd4b63760d1872b14524109089cf6aaf14105de24a2c3bc33ec1a822bcf77335` |
| `evidence/demos/demo-1-screenshot.png` | `b07accd28594a556edbd4aa0dbe5f39eeccfe01a88ced7420a42dcfda69c4f8b` |
| `evidence/demos/demo-2-annotated.png` | `8569485853d82c7f3b74817c06f91ae6e10eb5f054df0b8533838ef630d7d3b5` |
| `evidence/demos/demo-2-fixture-acceptance.json` | `48d42f5bc132461c1b5a5b2fecaa3f415f44d71ed07c33c16549f32fc17e9ba3` |
| `evidence/demos/demo-2-screenshot.png` | `7202437ab2c00aab8ee0ee55d99544af2c77c8a56c2496c431146e21caf584d6` |
| `evidence/demos/demo-2-session.json` | `ad840e57a21334ab84db4c09d6b1dd6db84dfa495369678c707358c382efd4b3` |

The corresponding operator-facing HTML is
`workbench/one-command-demos.html`.

## Clean-public verification

With all owner-supplied camera artifacts absent, the installed public test
environment completes with **352 passed and 180 explicitly skipped**. Skips
cover tests requiring a recovered model, the recovered CV22 forensic corpus,
generated ADES oracle bundles, or an unavailable optional accelerator.

With an owner-supplied horizontal model mounted temporarily, the exact Demo 2
production-detection regression completes successfully. The model is removed
again before packaging and is not present in the release history or wheel.

Add the intended public license before publishing this staging tree.
