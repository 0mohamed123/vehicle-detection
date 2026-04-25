from detector import VehicleDetector
from tracker import VehicleTracker


def evaluate():
    detector = VehicleDetector(model_size='n', conf=0.25)
    tracker = VehicleTracker()

    test_urls = [
        'https://ultralytics.com/images/bus.jpg',
        'https://ultralytics.com/images/zidane.jpg',
    ]

    print("=" * 55)
    print("   Vehicle Detection & Tracking System")
    print("=" * 55)

    total_vehicles = 0

    for url in test_urls:
        name = url.split('/')[-1]
        print(f"\nImage: {name}")

        detections = detector.detect(url)
        vehicle_dets = detections
        counts = detector.count_by_type(vehicle_dets)
        total_vehicles += len(vehicle_dets)

        print(f"Vehicles detected: {len(vehicle_dets)}")
        for cls, count in counts.items():
            confs = [d['confidence'] for d in vehicle_dets if d['class'] == cls]
            avg_conf = sum(confs) / len(confs) if confs else 0
            print(f"  {cls:12s}: {count} | avg conf: {avg_conf:.2f}")

        tracked = tracker.update(vehicle_dets)
        print(f"Active tracks: {len(tracked)}")

    print(f"\n{'='*55}")
    print(f"Total vehicles detected: {total_vehicles}")
    print(f"Tracker stats: {tracker.get_stats()}")
    print("=" * 55)


if __name__ == '__main__':
    evaluate()