"""
DetectorAgent — Node 1 in the VisionGraph.

Runs YOLOv8 inference on the incoming frame.
Falls back to a mock detector if ultralytics is not installed,
so the pipeline runs end-to-end even in environments without a GPU.
"""

import cv2
import numpy as np
import time
from typing import Any

try:
    from ultralytics import YOLO
    _model = YOLO("yolov8n.pt")   # nano — fast, ~6 MB
    YOLO_AVAILABLE = True
except Exception:
    _model = None
    YOLO_AVAILABLE = False


# Classes we care about for the event rules
TRACKED_CLASSES = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "knife", "scissors", "fire hydrant", "backpack", "suitcase"
}


class DetectorAgent:
    """
    Input  state keys : frame, timestamp
    Output state keys : detections (list of dicts)

    Each detection:
        {
            "class_name": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],   # absolute pixel coords
            "center": [cx, cy]
        }
    """

    def __init__(self):
        self.model = _model
        self.yolo_available = YOLO_AVAILABLE

    def run(self, state: dict) -> dict:
        frame: np.ndarray = state["frame"]
        t0 = time.perf_counter()

        if self.yolo_available and self.model is not None:
            detections = self._yolo_detect(frame)
        else:
            detections = self._mock_detect(frame)

        state["detections"] = detections
        state["_detector_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return state

    def _yolo_detect(self, frame: np.ndarray) -> list:
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            if class_name not in TRACKED_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < 0.35:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_name": class_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            })
        return detections

    def _mock_detect(self, frame: np.ndarray) -> list:
        """
        Deterministic mock: uses simple OpenCV HOG person detector
        so the pipeline is testable without YOLOv8.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(
            gray, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        detections = []
        for (x, y, bw, bh), w_score in zip(rects, weights):
            detections.append({
                "class_name": "person",
                "confidence": round(float(w_score[0]), 3),
                "bbox": [x, y, x + bw, y + bh],
                "center": [x + bw // 2, y + bh // 2],
                "source": "hog_fallback"
            })
        return detections
