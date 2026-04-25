from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time

VEHICLE_CLASSES = {
    2: 'car', 3: 'motorcycle', 5: 'bus',
    7: 'truck', 1: 'bicycle'
}

class VehicleDetector:
    def __init__(self, model_size='n', conf=0.25):
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.conf = conf
        self.vehicle_classes = VEHICLE_CLASSES

    def detect(self, source, save=False, output_dir='results'):
        results = self.model(source, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                if cls_id in self.vehicle_classes:
                    detections.append({
                        'class': self.vehicle_classes[cls_id],
                        'confidence': round(float(box.conf), 3),
                        'bbox': [round(x, 1) for x in box.xyxy[0].tolist()],
                        'class_id': cls_id
                    })
            if save:
                Path(output_dir).mkdir(exist_ok=True)
                r.save(filename=f"{output_dir}/detected_{Path(str(source)).name}")
        return detections

    def detect_url(self, url, conf=None):
        if conf:
            self.conf = conf
        return self.detect(url)

    def count_by_type(self, detections):
        counts = {}
        for d in detections:
            cls = d['class']
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def get_model_info(self):
        return {
            'model': f'YOLOv8{self.model.ckpt_path}',
            'vehicle_classes': list(self.vehicle_classes.values()),
            'total_classes': len(self.vehicle_classes)
        }

    def benchmark(self, url, n=10):
        times = []
        for _ in range(n):
            start = time.time()
            self.detect(url)
            times.append(time.time() - start)
        return {
            'avg_time': round(np.mean(times), 3),
            'min_time': round(np.min(times), 3),
            'max_time': round(np.max(times), 3),
            'fps': round(1 / np.mean(times), 1)
        }