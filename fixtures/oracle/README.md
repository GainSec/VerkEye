# ADES differential oracle bundles

Oracle bundles preserve one exact CB62 model input, all 22 split-boundary
tensor files, the assembled `13566x8 float32` prediction matrix, core
detections, and the identities of the digest-pinned ADES runtime that produced
them.

Large bundle members are generated under `.runtime/oracle/` and are not stored
in Git. The bundle manifest records every member size and SHA-256. Loading is
fail-closed: a missing, changed, extra-path, malformed, or numerically invalid
member is rejected before any differential result is exposed.

Generate the standard deterministic corpus with:

```bash
.venv/bin/python scripts/capture_ades_oracle.py \
  --runtime config/cb62-ades-runtime.json \
  --model fixtures/models/yolov6n_hor.bin \
  --pipeline evidence/compatibility/cvproc-yolov6-pipeline.json \
  --workspace .runtime/ades-cb62 \
  --output .runtime/oracle \
  --catalog fixtures/oracle/catalog.json
```

The standard cases are zero, impulse, ramp, deterministic random, and the
existing exact-size synthetic media fixture. Additional exact-size real images
may be supplied with `--image NAME=PATH`.
