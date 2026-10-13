"""Conservative short-track association; NOT cross-day identity recognition."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    box: np.ndarray
    confidence: float = 1.0
    appearance: np.ndarray = None

    @property
    def center(self):
        return (self.box[:2] + self.box[2:]) / 2

    @property
    def scale(self):
        return max(float(self.box[2] - self.box[0]), 1.0)


def appearance(frame, box):
    x1, y1, x2, y2 = np.asarray(box, int)
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return np.zeros(32, dtype=float)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    return hist / max(float(hist.sum()), 1)


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, time_s):
        self.tracks = {k: t for k, t in self.tracks.items() if time_s - t["time"] <= self.cfg["max_gap_seconds"]}
        ids = list(self.tracks)
        cost = np.full((len(ids), len(detections)), 1e6)
        for i, tid in enumerate(ids):
            tr = self.tracks[tid]
            predicted = tr["detection"].center + tr["velocity"] * (time_s - tr["time"])
            for j, det in enumerate(detections):
                dist = np.linalg.norm(predicted - det.center) / max(tr["detection"].scale, det.scale)
                ratio = det.scale / tr["detection"].scale
                if dist > self.cfg["match_gate_body_lengths"] or not 0.4 < ratio < 2.5:
                    continue
                old_app, new_app = tr["detection"].appearance, det.appearance
                app_cost = 0 if old_app is None or new_app is None else np.abs(old_app-new_app).sum() / 2
                cost[i, j] = dist + 0.8 * app_cost
        matches = {}
        if cost.size:
            ii, jj = linear_sum_assignment(cost)
            for i, j in zip(ii, jj):
                if cost[i, j] >= 1e5:
                    continue
                # Ambiguous crossings start new tracklets instead of inventing certainty.
                row = np.sort(cost[i]); col = np.sort(cost[:, j])
                ambiguous = (len(row) > 1 and row[1]-row[0] < 0.08) or (len(col) > 1 and col[1]-col[0] < 0.08)
                if not ambiguous:
                    matches[j] = ids[i]
        output = []
        for j, det in enumerate(detections):
            tid = matches.get(j)
            associated = tid is not None
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                velocity = np.zeros(2)
            else:
                old = self.tracks[tid]
                dt = time_s - old["time"]
                velocity = (det.center-old["detection"].center) / max(dt, 1e-6)
            self.tracks[tid] = {"detection": det, "velocity": velocity, "time": time_s}
            output.append((tid, det, associated))
        return output


class MotionDetector:
    """Experimental foreground blobs: still fish can disappear; not a fish classifier."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)
        self.seen = 0

    def detect(self, frame):
        self.seen += 1
        mask = self.bg.apply(frame, learningRate=0.01)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if self.seen < 5:
            return []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.cfg["min_detection_area"] <= area <= frame.shape[0]*frame.shape[1]*self.cfg["max_detection_area_fraction"]:
                x, y, w, h = cv2.boundingRect(c)
                box = np.array([x, y, x+w, y+h], dtype=float)
                result.append(Detection(box, 0.5, appearance(frame, box)))
        return result


class YoloDetector:
    def __init__(self, weights, cfg, class_id=0):
        from pathlib import Path
        if not Path(weights).is_file():
            raise ValueError("需要已存在的本地神仙鱼检测权重；不会自动下载通用模型冒充鱼类模型")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ValueError('请先安装检测扩展: pip install -e ".[detector]"') from exc
        self.model, self.cfg, self.class_id = YOLO(weights), cfg, class_id

    def detect(self, frame):
        result = self.model.predict(frame, conf=self.cfg["min_detection_confidence"], classes=[self.class_id], verbose=False)[0]
        if result.boxes is None:
            return []
        return [Detection(box, float(conf), appearance(frame, box)) for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]
