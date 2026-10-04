#!/usr/bin/env python3
"""Run the exact accelerated CB62 viewer headlessly on prerecorded media."""

from __future__ import annotations

import argparse
from pathlib import Path

from verkeye.accelerated.cb62 import AcceleratedCb62Runtime, Cb62MlxSession
from verkeye.compat.ades_runtime import load_runtime_spec
from verkeye.runtime.inference import ExactFrameSession
from verkeye.runtime.sources import iter_capture
from verkeye.viewer.controller import LiveViewerSession, open_video_writer
from verkeye.viewer.smoke import HeadlessDisplay, resolve_smoke_video


REPOSITORY = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=REPOSITORY / "fixtures/models/yolov6n_hor.bin",
    )
    parser.add_argument(
        "--runtime-spec",
        type=Path,
        default=REPOSITORY / "config/cb62-ades-runtime.json",
    )
    parser.add_argument(
        "--pipeline-evidence",
        type=Path,
        default=REPOSITORY
        / "evidence/compatibility/cvproc-yolov6-pipeline.json",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-full-kernels",
    )
    parser.add_argument(
        "--split4-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-split4",
    )
    parser.add_argument(
        "--split5-root",
        type=Path,
        default=REPOSITORY / ".runtime/ades-split5",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="existing video; omit to regenerate the deterministic fixture",
    )
    parser.add_argument(
        "--backend",
        choices=("accelerated",),
        default="accelerated",
        help="exact local runtime (only accelerated is supported)",
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPOSITORY / "evidence/viewer/macos-prerecorded-summary.json",
    )
    parser.add_argument(
        "--frame-out",
        type=Path,
        default=REPOSITORY / "evidence/viewer/macos-prerecorded-frame.png",
    )
    parser.add_argument("--record", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    video = resolve_smoke_video(
        requested=args.video,
        generated=REPOSITORY / ".runtime/viewer-smoke-input.avi",
        frame_count=args.frames,
    )
    spec = load_runtime_spec(args.runtime_spec)
    backend = AcceleratedCb62Runtime(
        session=Cb62MlxSession(
            capture_root=args.capture_root,
            split4_root=args.split4_root,
            split5_root=args.split5_root,
        )
    )
    exact = ExactFrameSession(
        model=args.model,
        backend=backend,
        runtime_spec=spec,
        pipeline_evidence=args.pipeline_evidence,
    )
    display = HeadlessDisplay()
    writer = None
    recording = None
    if args.record is not None:
        writer = open_video_writer(args.record, fps=30.0)
        recording = {
            "path": str(args.record.resolve()),
            "codec": "mp4v",
            "fps": 30.0,
            "size": [608, 1088],
        }
    viewer = LiveViewerSession(
        source=iter_capture(video),
        exact_session=exact,
        display=display,
        source_label=f"video {video}",
        run_id="macos-prerecorded-smoke",
        writer=writer,
        recording=recording,
        summary_path=args.summary_out,
        max_frames=args.frames,
    )
    summary = viewer.run()
    display.write_last_frame(args.frame_out)
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.frame_out}")
    print(f'inference_fps={summary["metrics"]["inference_fps"]:.6f}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
