# VerkEye public release manifest

Public project: <https://github.com/GainSec/VerkEye>

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

## Owner-generated runtime boundary

The repository includes the independently authored generator, normalized
runtime format, manifest verifier, native capture hooks, and accelerated
execution implementation. It does not include generated runtime parameters,
weights, masks, raw ADES captures, or preparation workspaces.

Owners can run `verkeye generate-runtime` against the hash-pinned model
extracted from their own CB62. Output is written beneath the Git-ignored
`.runtime/generated/` tree and is verified before atomic installation. Public
wheels include generator source and capture tooling, not the owner model or
the resulting runtime contents.

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

## Evidence boundary

The public `evidence/` tree is intentionally limited to the runtime pipeline
contract, the published demo outputs, and compact machine-readable parity and
performance records. Large intermediate traces, duplicated captures, raw
development runs, and private forensic working material are excluded from the
public repository because they are not needed to install, run, or audit the
published claims. The exact retained inventory and purpose of each group are
documented in `evidence/README.md`.

## Clean-public verification

With all owner-supplied camera artifacts absent, the installed public test
environment completes with **336 passed and 198 explicitly skipped**. Skips
cover tests requiring a recovered model, the recovered CV22 forensic corpus,
generated ADES oracle bundles, or an unavailable optional accelerator.

With an owner-supplied horizontal model mounted temporarily, the exact Demo 2
production-detection regression completes successfully. The model is removed
again before packaging and is not present in the release history or wheel.

Add the intended public license before publishing this staging tree.
