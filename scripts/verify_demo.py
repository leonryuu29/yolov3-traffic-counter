"""Load the YOLO model and run one inference without opening a GUI window."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "yolov3.weights")
    parser.add_argument("--config", type=Path, default=ROOT / "yolov3.cfg")
    parser.add_argument("--video", type=Path, default=ROOT / "traffic_video.mp4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for label, path in (
        ("weights", args.weights),
        ("config", args.config),
        ("video", args.video),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    net = cv2.dnn.readNet(str(args.weights), str(args.config))
    capture = cv2.VideoCapture(str(args.video))
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read a frame from {args.video}")

        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
    finally:
        capture.release()

    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")
    print(f"YOLO output tensors: {[tuple(output.shape) for output in outputs]}")
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
