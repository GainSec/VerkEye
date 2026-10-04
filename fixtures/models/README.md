# Recovered model fixtures

The recovered model files are owner-supplied research artifacts. They are not
interchangeable with public YOLO models and are not distributed in this
repository.

Extract `yolov6n_hor.bin` from a CB62 you own, place it in this directory, and
verify its size and SHA-256 before exact inference. The generated VerkEye
runtime remains part of the project and is not a separate asset bundle. The
vertical comparison model is optional and needed only for forensic comparison.

| File | Role | Size | SHA-256 |
|---|---|---:|---|
| `yolov6n_hor.bin` | Production horizontal CB62 model | 5,561,124 | `eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf` |
| `yolov6n_ver.bin` | Same-device vertical-model comparison oracle | 5,551,804 | `e19ca243214332ae4bd1cec2e828366f66860d10f49e069c3f376941c92ec129` |

Neither file in the table is included. The comparison oracle is used only to
identify byte-identical compiled regions.
Identity across compiler outputs does not prove whether a region is code,
weights, constants, relocation material, or some other accelerator input.
