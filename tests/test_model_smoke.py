from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "yolov3.weights"
CONFIG = ROOT / "yolov3.cfg"
VIDEO = ROOT / "traffic_video.mp4"


@unittest.skipUnless(
    WEIGHTS.exists() and CONFIG.exists() and VIDEO.exists(),
    "Model weights and a local sample video are required",
)
class ModelSmokeTests(unittest.TestCase):
    def test_yolo_model_loads_and_processes_one_frame(self) -> None:
        net = cv2.dnn.readNet(str(WEIGHTS), str(CONFIG))
        cap = cv2.VideoCapture(str(VIDEO))
        self.addCleanup(cap.release)

        ok, frame = cap.read()
        self.assertTrue(ok)

        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        self.assertGreaterEqual(len(outputs), 1)
        self.assertTrue(all(isinstance(output, np.ndarray) for output in outputs))


if __name__ == "__main__":
    unittest.main()
