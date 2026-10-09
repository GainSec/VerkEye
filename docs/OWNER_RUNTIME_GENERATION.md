# Owner-local runtime generation

VerkEye does not distribute CB62 model packages, extracted firmware, or
generated model parameters. The `generate-runtime` command derives the compact
accelerated runtime on the owner's machine from a supported model extracted
from hardware they control.

## Requirements

- Python 3.11 or newer and an editable or wheel installation of VerkEye.
- Docker, Podman, or another compatible command capable of running a pinned
  `linux/amd64` image. Apple-silicon hosts require amd64 emulation support.
- An owner-supplied `yolov6n_hor.bin` extracted from a CB62.
- Approximately 1 GB of temporary free space. The temporary ADES workspace is
  substantially larger than the normalized runtime.

The currently supported model contract is:

```text
size     5,561,124 bytes
sha256   eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf
```

Other model hashes fail closed. This does not imply that other CB62 revisions
are incompatible; it means their generated graphs and parameters have not been
validated against this runtime contract.

## Generate

From a fresh checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,runtime,macos]'   # Apple silicon / MLX
# Linux users select '.[dev,runtime,linux]' instead.

mkdir -p fixtures/models
cp /path/to/owner-extracted/yolov6n_hor.bin fixtures/models/yolov6n_hor.bin

verkeye generate-runtime fixtures/models/yolov6n_hor.bin
```

For Podman or a Docker wrapper:

```bash
verkeye generate-runtime fixtures/models/yolov6n_hor.bin \
  --docker-command podman
```

The default destination is content-addressed:

```text
.runtime/generated/eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf/
```

The command performs these gates before installation:

1. verifies the owner model's SHA-256;
2. verifies or downloads the digest-pinned ADES container image;
3. regenerates and verifies all ten DVI files;
4. compiles the native executor and two capture hooks locally;
5. executes one bounded zero-input inference;
6. requires exactly 55 valid fast-convolution captures;
7. requires exactly one mask for each proved spatial contract;
8. removes process-local pointers and unused packed-row padding bits;
9. records every normalized member's size and SHA-256; and
10. validates the staged manifest before atomically installing it.

A failed run never installs a partial runtime. Re-running against an existing
destination fails unless `--force` is supplied. `--force` replaces only the
content-addressed directory for the verified model.

## Run

Once generated, the normal commands discover the runtime automatically:

```bash
verkeye run fixtures/models/yolov6n_hor.bin \
  --image FRAME.png --backend accelerated \
  --json-out RESULT.json --manifest-out RESULT.manifest.json

verkeye live fixtures/models/yolov6n_hor.bin \
  --webcam 0 --backend accelerated
```

For a relocated runtime parent, add:

```text
--generated-runtime-base /path/to/generated
```

For the fixed demos, set both locations when they differ from repository
defaults:

```bash
export VERKEYE_MODEL=/path/to/yolov6n_hor.bin
export VERKEYE_RUNTIME_BASE=/path/to/generated
verkeye --demo 2
```

## Diagnostics and cleanup

Supply `--workspace PATH` to retain the full ADES preparation and raw capture
for local debugging, or use `--keep-workspace` to retain it at the path printed
in the result. Treat that workspace as model-derived private material; do not
attach it to a public issue. Without either option, temporary capture data is
removed after successful or failed generation.

The normalized runtime is also model-derived and remains ignored by Git. To
remove it, delete only its exact content-addressed directory. The owner model
and any generated runtime should be reviewed under the owner's applicable
authorization and license terms; this project does not assert legal clearance
for third-party artifacts or tooling.
