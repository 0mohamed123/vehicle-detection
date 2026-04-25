# Vehicle Detection & Tracking System

![Language](https://img.shields.io/badge/Language-Python-blue)
![Model](https://img.shields.io/badge/Model-YOLOv8-purple)
![Tests](https://img.shields.io/badge/Tests-11%20passing-green)

Real-time vehicle detection and tracking system using YOLOv8.
Detects cars, buses, trucks, motorcycles, and bicycles with confidence scores and bounding boxes.

## Detection Results

    Image: bus.jpg
      bus     : 1 vehicle | avg conf: 0.87
    Active tracks: 1

    Total vehicles detected: 1
    Tracker stats: total_tracked=1, active_tracks=1

## Quick Start

    git clone https://github.com/0mohamed123/vehicle-detection.git
    cd vehicle-detection
    pip install ultralytics opencv-python

    cd src
    python evaluate.py

    cd ../tests
    python -m pytest test_detector.py -v

## Vehicle Classes

    car, bus, truck, motorcycle, bicycle

## Features

- YOLOv8 nano model for fast inference
- Filters only vehicle classes from 80 COCO classes
- Simple centroid-based multi-object tracker
- Consistent track IDs across frames
- Confidence threshold control
- Batch detection support

## Test Results

    11 passed | 0 failed

    Tests cover: model loading, vehicle classes, URL detection,
    bus detection, detection format, count by type,
    vehicles only filter, empty tracker, new tracks,
    consistent IDs, tracker stats

## Technologies

- Python 3.12
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- pytest (11 tests)