"""Download the official YOLOv3 weights with an integrity check."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://pjreddie.com/media/files/yolov3.weights"
SHA256 = "523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297"
DESTINATION = Path(__file__).resolve().parents[1] / "yolov3.weights"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if DESTINATION.exists():
        actual = sha256(DESTINATION)
        if actual == SHA256:
            print(f"Already downloaded and verified: {DESTINATION}")
            return 0
        print(f"Existing file has the wrong SHA-256: {actual}", file=sys.stderr)
        print("Move it elsewhere or remove it explicitly, then rerun.", file=sys.stderr)
        return 2

    partial = DESTINATION.with_suffix(DESTINATION.suffix + ".part")
    try:
        with urllib.request.urlopen(URL, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            with partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = downloaded * 100 / total
                        print(f"\rDownloading: {percent:5.1f}%", end="", flush=True)
        print()

        actual = sha256(partial)
        if actual != SHA256:
            partial.unlink(missing_ok=True)
            print(f"Checksum mismatch: expected {SHA256}, got {actual}", file=sys.stderr)
            return 1

        partial.replace(DESTINATION)
        print(f"Verified and saved: {DESTINATION}")
        return 0
    except Exception as exc:
        partial.unlink(missing_ok=True)
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
