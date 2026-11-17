# Third-party notices

## SORT

`sort.py` is based on the Simple Online and Realtime Tracking implementation by
Alex Bewley and contributors:

- Upstream: <https://github.com/abewley/sort>
- License: GNU General Public License v3.0

The repository therefore uses GPL-3.0. See [`LICENSE`](LICENSE).

## YOLOv3

The project uses the YOLOv3 network configuration and official pre-trained
weights distributed by the Darknet/YOLO authors:

- Paper: Redmon and Farhadi, *YOLOv3: An Incremental Improvement* (2018)
- Weight download: <https://pjreddie.com/media/files/yolov3.weights>
- Configuration reference: <https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg>

Model weights are downloaded by `scripts/download_models.py` and are not
committed to this repository.

## Open-source runtime dependencies

OpenCV, NumPy, SciPy, and the optional ONNX Runtime/PyTorch packages remain under
their respective upstream licenses.
