import numpy as np


class VehicleTracker:
    def __init__(self, max_distance=100):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = max_distance

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, c1, c2):
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def update(self, detections):
        if not detections:
            return {}

        centers = [self._center(d['bbox']) for d in detections]
        updated = {}

        if not self.tracks:
            for i, det in enumerate(detections):
                self.tracks[self.next_id] = {
                    'center': centers[i],
                    'class': det['class'],
                    'count': 1
                }
                updated[self.next_id] = det
                self.next_id += 1
            return updated

        used_tracks = set()
        used_dets = set()

        for i, center in enumerate(centers):
            best_id = None
            best_dist = self.max_distance
            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                dist = self._distance(center, track['center'])
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                self.tracks[best_id]['center'] = center
                self.tracks[best_id]['count'] += 1
                updated[best_id] = detections[i]
                used_tracks.add(best_id)
                used_dets.add(i)

        for i, det in enumerate(detections):
            if i not in used_dets:
                self.tracks[self.next_id] = {
                    'center': centers[i],
                    'class': det['class'],
                    'count': 1
                }
                updated[self.next_id] = det
                self.next_id += 1

        return updated

    def get_stats(self):
        return {
            'total_tracked': len(self.tracks),
            'active_tracks': self.next_id - 1
        }