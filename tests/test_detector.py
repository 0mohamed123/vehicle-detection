import sys
sys.path.append('../src')

import numpy as np
import pytest
from detector import VehicleDetector, VEHICLE_CLASSES
from tracker import VehicleTracker


@pytest.fixture(scope='module')
def detector():
    return VehicleDetector(model_size='n', conf=0.25)


# ===== Detector Tests =====
def test_model_loads(detector):
    assert detector.model is not None


def test_vehicle_classes():
    assert 2 in VEHICLE_CLASSES
    assert VEHICLE_CLASSES[2] == 'car'
    assert VEHICLE_CLASSES[5] == 'bus'
    assert len(VEHICLE_CLASSES) == 5


def test_detect_url(detector):
    detections = detector.detect_url('https://ultralytics.com/images/bus.jpg')
    assert isinstance(detections, list)


def test_detect_bus(detector):
    detections = detector.detect_url('https://ultralytics.com/images/bus.jpg')
    classes = [d['class'] for d in detections]
    assert 'bus' in classes


def test_detection_format(detector):
    detections = detector.detect_url('https://ultralytics.com/images/bus.jpg')
    for d in detections:
        assert 'class' in d
        assert 'confidence' in d
        assert 'bbox' in d
        assert 0 <= d['confidence'] <= 1
        assert len(d['bbox']) == 4


def test_count_by_type(detector):
    detections = detector.detect_url('https://ultralytics.com/images/bus.jpg')
    counts = detector.count_by_type(detections)
    assert isinstance(counts, dict)
    assert counts.get('bus', 0) >= 1


def test_only_vehicles(detector):
    detections = detector.detect_url('https://ultralytics.com/images/bus.jpg')
    for d in detections:
        assert d['class'] in VEHICLE_CLASSES.values()


# ===== Tracker Tests =====
def test_tracker_empty():
    tracker = VehicleTracker()
    result = tracker.update([])
    assert result == {}


def test_tracker_new_tracks():
    tracker = VehicleTracker()
    dets = [
        {'class': 'car', 'confidence': 0.9, 'bbox': [10, 10, 50, 50], 'class_id': 2},
        {'class': 'bus', 'confidence': 0.8, 'bbox': [100, 100, 200, 200], 'class_id': 5},
    ]
    result = tracker.update(dets)
    assert len(result) == 2


def test_tracker_consistent_ids():
    tracker = VehicleTracker()
    det = [{'class': 'car', 'confidence': 0.9,
            'bbox': [10, 10, 50, 50], 'class_id': 2}]
    result1 = tracker.update(det)
    det2 = [{'class': 'car', 'confidence': 0.9,
             'bbox': [12, 12, 52, 52], 'class_id': 2}]
    result2 = tracker.update(det2)
    assert list(result1.keys())[0] == list(result2.keys())[0]


def test_tracker_stats():
    tracker = VehicleTracker()
    dets = [{'class': 'car', 'confidence': 0.9,
             'bbox': [10, 10, 50, 50], 'class_id': 2}]
    tracker.update(dets)
    stats = tracker.get_stats()
    assert stats['total_tracked'] == 1
    assert stats['active_tracks'] == 1