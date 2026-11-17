from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_models.py"
SPEC = importlib.util.spec_from_file_location("download_models", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloadModelTests(unittest.TestCase):
    def test_sha256_matches_known_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "sample.bin"
            target.write_bytes(b"yolov3")
            self.assertEqual(
                MODULE.sha256(target),
                "7bbde3ae22df7f4dc70ec6a50ebed8c1fd115f540ac9190ebc0aba5157e17c69",
            )


if __name__ == "__main__":
    unittest.main()
