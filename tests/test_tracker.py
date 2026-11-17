from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sort import iou_batch, linear_assignment  # noqa: E402


class TrackerUtilityTests(unittest.TestCase):
    def test_iou_batch_returns_expected_values(self) -> None:
        detections = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
        tracks = np.array([[5, 0, 15, 10]], dtype=float)

        overlaps = iou_batch(detections, tracks)

        self.assertEqual(overlaps.shape, (2, 1))
        self.assertAlmostEqual(float(overlaps[0, 0]), 1 / 3, places=6)
        self.assertEqual(float(overlaps[1, 0]), 0.0)

    def test_linear_assignment_uses_minimum_total_cost(self) -> None:
        cost = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)

        matches = linear_assignment(cost)

        self.assertEqual(
            sorted((int(row), int(track)) for row, track in matches),
            [(0, 1), (1, 0)],
        )


if __name__ == "__main__":
    unittest.main()
