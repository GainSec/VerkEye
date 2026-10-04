# Public evidence boundary

This directory contains only evidence required to run VerkEye from a source
checkout or to audit the principal public demo, equivalence, and performance
claims. It deliberately excludes the much larger raw research corpus,
intermediate traces, repeated captures, and recovered proprietary artifacts.

## Runtime contract

- `compatibility/cvproc-yolov6-pipeline.json` — the integrity-pinned CB62
  preprocessing, post-processing, class-map, and detector ABI contract used by
  the source checkout and copied into installed packages.

## Published demos

- `demos/demo-1-live-smoke.json`
- `demos/demo-1-screenshot.png`
- `demos/demo-2-annotated.png`
- `demos/demo-2-fixture-acceptance.json`
- `demos/demo-2-screenshot.png`
- `demos/demo-2-session.json`
- `public-webcams/data-gov-sg-20261003/results/camera-2704-small-object.png`
- `public-webcams/small-object-validation-20261003.json`

These are the machine-readable records and rendered examples referenced by the
README and VerkEye Workbench.

## Exactness and performance summaries

- `equivalence/fresh-clone-verification.txt`
- `equivalence/kali-openvino-terminal-parity.json`
- `equivalence/macos-mlx-terminal-parity.json`
- `equivalence/production-detection-parity.json`
- `performance/kali-openvino-equivalent-final.json`
- `performance/macos-mlx-equivalent-final.json`
- `final-verification.txt`

These compact records preserve the public parity and throughput results. The
distribution test in `tests/test_distribution.py` enforces this exact inventory
so unrelated or sensitive research material cannot enter a public release by
accident.

## Excluded owner artifact

`yolov6n_hor.bin` is not distributed. Owners must extract it from a CB62 they
own and verify its documented SHA-256 before use. The public release contains
no recovered Verkada model, CV22 executable, library, driver, firmware,
microcode, or production configuration.
