# YOLOv3-based Traffic Detection System - Technical Documentation

This document provides a technical overview of the vehicle detection and tracking system, which consists of two main Python scripts: `traffic_detector.py` and `sort.py`.

## 1. `traffic_detector.py` - Vehicle Detection and Tracking

### 1.1. Overview

`traffic_detector.py` is the main script that performs real-time vehicle detection, tracking, speed estimation, and counting from a video source. It uses the YOLOv3 (You Only Look Once) object detection model to identify vehicles and the SORT (Simple Online and Realtime Tracking) algorithm to track them across frames.

### 1.2. Dependencies

-   OpenCV (`cv2`)
-   NumPy (`numpy`)

The script also relies on `sort.py` for the tracking functionality.

### 1.3. Configuration Parameters

The script's behavior can be configured through several parameters at the beginning of the file:

-   `VIDEO_SOURCE`: Path to the input video file.
-   `WEIGHTS_PATH`: Path to the YOLOv3 model weights file (`.weights`).
-   `CONFIG_PATH`: Path to the YOLOv3 model configuration file (`.cfg`).
-   `NAMES_PATH`: Path to the file containing the names of the classes the YOLOv3 model can detect.
-   `CONFIDENCE_THRESHOLD`: The minimum confidence score for a detection to be considered.
-   `NMS_THRESHOLD`: The threshold for non-maximum suppression, used to eliminate redundant overlapping bounding boxes.
-   `HIGH_CONFIDENCE_THRESHOLD`: A threshold to mark a tracked vehicle as "high confidence". This is used to ensure only reliable tracks are counted.
-   `SORT_MAX_AGE`: The maximum number of frames a track can be lost before it is deleted.
-   `SORT_MIN_HITS`: The minimum number of consecutive detections required to start a new track.
-   `PIXELS_PER_METER`: A calibration parameter to convert pixel distances to meters for speed estimation. This needs to be adjusted for each specific camera setup.
-   `LINE_Y`: The Y-coordinate of the virtual line used for counting vehicles.
-   `FRAME_SKIP`: Process every Nth frame to improve performance.

### 1.4. Workflow

The script processes the video stream frame by frame in a loop:

1.  **Frame Reading**: Reads a frame from the video source.
2.  **Object Detection**:
    -   The frame is pre-processed and fed into the YOLOv3 network.
    -   The network outputs bounding boxes, confidence scores, and class IDs for detected objects.
    -   Detections are filtered based on `CONFIDENCE_THRESHOLD` and if the class is a vehicle (car, truck, bus, motorbike).
3.  **Non-Maximum Suppression (NMS)**: `cv2.dnn.NMSBoxes` is used to suppress weak and overlapping bounding boxes.
4.  **Tracking**:
    -   The filtered detections are passed to the `Sort` tracker instance (`mot_tracker`).
    -   The tracker's `update()` method returns a list of currently tracked objects, each with a unique ID.
5.  **Data Association and Information Update**:
    -   For each tracked object, the script finds the best-matching detection from the current frame using the Intersection over Union (IoU) metric.
    -   Information like the vehicle's class label, confidence, and color for visualization is updated and stored.
6.  **Speed Estimation**:
    -   The script calculates the speed of each tracked vehicle based on the displacement of its center point between frames. The `get_speed` function is used for this.
7.  **Vehicle Counting**:
    -   The script checks for tracks that have disappeared (`lost_track_ids`).
    -   If a lost track was marked as "high confidence" and its path crossed the counting line (`LINE_Y`), the vehicle count is incremented.
8.  **Visualization**:
    -   The script draws bounding boxes, track IDs, labels, confidence scores, and estimated speeds on the frame.
    -   The counting line and the total vehicle count are also displayed.
9.  **Display**: The processed frame is shown in a window.

The loop terminates when the video ends or the user presses the 'ESC' key.

### 1.5. Key Functions

-   `get_speed(prev_center, current_center, time_interval)`: Estimates the speed of a vehicle in km/h by calculating the distance traveled in meters (converted from pixels) over a given time interval.
-   `iou(boxA, boxB)`: Calculates the Intersection over Union (IoU) of two bounding boxes. This is used to associate detections with tracks.

## 2. `sort.py` - Simple Online and Realtime Tracking (SORT)

### 2.1. Overview

`sort.py` provides the implementation of the SORT algorithm, a simple yet effective approach for online and real-time object tracking. It's designed to be used with an external object detector.

### 2.2. Core Components

The implementation consists of three main classes:

-   `KalmanFilter`: A simple implementation of a linear Kalman filter. It is used to predict the future location of a tracked object and to smooth the noisy detections.
-   `KalmanBoxTracker`: Represents a single tracked object. It wraps the `KalmanFilter` and manages the state of an object (e.g., its age, number of hits, etc.).
-   `Sort`: The main class that manages all the `KalmanBoxTracker` instances. It handles the association of new detections to existing trackers and the creation/deletion of tracks.

### 2.3. Algorithm

The SORT algorithm follows these steps in each frame:

1.  **Prediction**: For each existing track, the `KalmanFilter` predicts its new state (bounding box).
2.  **Association**: The predicted bounding boxes of the existing tracks are matched with the new detections from the object detector. The association is based on the IoU metric. The Hungarian algorithm (`linear_sum_assignment` from `scipy.optimize`) is used to find the optimal assignment that maximizes the IoU.
3.  **Update**:
    -   **Matched Tracks**: For each track that is successfully matched with a detection, its `KalmanFilter` is updated with the new measurement (the detected bounding box).
    -   **Unmatched Detections**: New `KalmanBoxTracker` instances are created for detections that could not be associated with any existing track.
    -   **Unmatched Tracks**: Tracks that are not matched with any detection are marked as "lost". If a track is lost for more than `max_age` frames, it is deleted.

### 2.4. Key Classes and Methods

#### `KalmanFilter`

-   `predict()`: Predicts the next state of the system.
-   `update(z)`: Updates the state based on a new measurement `z`.

#### `KalmanBoxTracker`

-   `__init__(bbox)`: Initializes a new tracker with an initial bounding box.
-   `predict()`: Advances the state vector and returns the predicted bounding box.
-   `update(bbox)`: Updates the tracker with a new bounding box measurement.
-   `get_state()`: Returns the current estimated bounding box.

#### `Sort`

-   `__init__(max_age=1, min_hits=3, iou_threshold=0.3)`: Initializes the tracker with configurable parameters.
-   `update(dets)`: The main method that takes the detections for the current frame and returns the updated tracks.
-   `associate_detections_to_trackers(detections, trackers, iou_threshold)`: Performs the association logic between detections and trackers.
