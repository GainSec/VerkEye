# Owner-supplied fixture boundary

The public repository intentionally excludes recovered camera artifacts,
including both model binaries and the CV22 vendor executables, libraries,
kernel module, firmware, bytecode, and production configuration. Extract
`models/yolov6n_hor.bin` from a CB62 you own and place it at that path before
running exact inference.
`scripts/extract_exact_model.sh` can extract the file from an owner-created
application-filesystem corpus.

- Expected size: `5,561,124` bytes
- Expected SHA-256:
  `eb768eb6691c08a5648424bb3fbc4a5ee09a64584e7faa3236afa778f5f5eabf`
Do not substitute public YOLO weights, zero-filled data, or another CB62 model
variant. VerkEye's generated accelerated runtime is included in this project;
the original device artifacts remain owner-supplied.
