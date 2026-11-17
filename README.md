# YOLOv3 Traffic Counter

**Real-time vehicle detection, tracking, and line-crossing counts with YOLOv3 + SORT.**

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](#quick-start)
[![OpenCV](https://img.shields.io/badge/OpenCV-DNN-5C3EE8?logo=opencv&logoColor=white)](#quick-start)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

> 一个基于 YOLOv3 与 SORT 的车辆检测、跟踪、计数和速度估算演示项目。<br>
> A computer-vision coursework project rebuilt into a reproducible, documented repository.

## Why this project is interesting

The pipeline is deliberately small enough to inspect end to end:

```mermaid
flowchart LR
    A[Video frame] --> B[YOLOv3 / OpenCV DNN]
    B --> C[Vehicle-class filter]
    C --> D[Non-maximum suppression]
    D --> E[SORT tracker]
    E --> F[Trajectory and speed]
    F --> G[Virtual-line counter]
    G --> H[Annotated live view]
```

It combines:

- YOLOv3 inference through OpenCV's DNN backend.
- SORT tracking with a Kalman filter and Hungarian assignment.
- Vehicle filtering for cars, trucks, buses, and motorbikes.
- Direction-aware virtual-line counting.
- Pixel-to-metre speed estimation.
- CPU execution and optional CUDA/ONNX experiments.

The included sample run counted **3 cars** in one short traffic clip; the raw output is preserved in [`docs/EXAMPLE_OUTPUT.txt`](docs/EXAMPLE_OUTPUT.txt). This is a demo result, not a benchmark across datasets.

## Quick start

### 1. Requirements

- Python 3.10–3.12
- A local video file
- Internet access for the one-time YOLOv3 weight download

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

### 2. Download the model weights

```bash
python scripts/download_models.py
```

The script downloads the official YOLOv3 weights and verifies the SHA-256 hash. The 248 MB weight file is intentionally excluded from Git history.

### 3. Run

Place the input clip at `traffic_video.mp4`, then run:

```bash
python traffic_detector.py
```

Press `Esc` to close the live window. The current entry point uses the filenames configured near the top of `traffic_detector.py`:

| Setting | Default |
|---|---|
| Video | `traffic_video.mp4` |
| Weights | `yolov3.weights` |
| Network config | `yolov3.cfg` |
| Class names | `coco.names` |
| Counting line | `LINE_Y = 400` |
| Pixel scale | `PIXELS_PER_METER = 70` |

Change the pixel scale for your camera and scene before interpreting the displayed speed.

## Repository layout

```text
.
├── traffic_detector.py          # Main CPU/OpenCV DNN pipeline
├── sort.py                      # Vendored SORT implementation
├── yolov3.cfg                   # YOLOv3 network structure
├── coco.names                   # COCO class labels
├── scripts/
│   └── download_models.py       # Verified weight download
├── experiments/
│   ├── traffic_detector_gpu.py  # ONNX Runtime experiment
│   └── build_onnx_from_local.py # Local ONNX export experiment
├── docs/
│   ├── RESEARCH_PAPER.md
│   ├── TECHNICAL_DOCUMENTATION.md
│   ├── CORE_IMPROVEMENTS.tex
│   └── EXAMPLE_OUTPUT.txt
└── tests/
    └── test_tracker.py
```

## Run the focused tests

```bash
python -m unittest discover -s tests -v
```

The tests cover IoU geometry and Hungarian assignment behavior in the vendored SORT utilities.

After downloading the weights and supplying a local video, run a headless one-frame
model check with:

```bash
python scripts/verify_demo.py
```

It reports the input frame dimensions and YOLO output tensor shapes without opening
a GUI window.

## Optional GPU / ONNX path

The experimental scripts are kept for context but are not required for the main demo:

```bash
python -m pip install -r requirements-onnx.txt
python experiments/traffic_detector_gpu.py
```

`experiments/build_onnx_from_local.py` documents an earlier ONNX export attempt. Treat it as an experiment rather than a supported production path.

## Design notes and limitations

- `PIXELS_PER_METER` is scene-specific; speed values are only rough estimates without homography calibration.
- The counting line and class filter are intentionally simple and should be adapted to the camera angle.
- YOLOv3 is retained because this was the original coursework model. Modern YOLO variants can replace it, but that would change the reproducibility baseline and dependency footprint.
- The input video and YOLO weight file are not redistributed here. Bring your own legally usable footage.
- Real-time FPS depends on CPU/GPU, video resolution, frame skipping, and OpenCV build.

## Documentation

- [Technical documentation](docs/TECHNICAL_DOCUMENTATION.md)
- [Research paper](docs/RESEARCH_PAPER.md)
- [Core improvements](docs/CORE_IMPROVEMENTS.tex)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

The repository is distributed under **GPL-3.0** because the vendored SORT implementation is GPL-3.0 software. YOLOv3 model files are downloaded separately from their upstream source and are not included in this repository.

---

**中文简介：** 这是一个使用 YOLOv3 和 SORT 完成车辆检测、跟踪、越线计数与速度估算的计算机视觉项目。仓库保留原始课程成果，并补充了模型下载校验、依赖说明、文档索引和可运行步骤。
