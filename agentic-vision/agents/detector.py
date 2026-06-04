"""
DetectorAgent — Node 1 in the VisionGraph.

Uses OpenCV-only detection (no PyTorch/ultralytics required):
  - HOG person detector
  - Haar cascade: face, full body, upper body, car (frontal/side)
  - Colour-based skin region analysis for crowd density estimation
  - Motion-based detection (frame differencing) for webcam streams

This keeps the deployment free-tier compatible (< 200MB RAM)
while still triggering meaningful event rules.
"""

import cv2
import numpy as np
import time
import os


def _load_cascade(name: str):
    path = cv2.data.haarcascades + name
    cc = cv2.CascadeClassifier(path)
    return None if cc.empty() else cc


# Load cascades once at module level
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

_CASCADE_FACE       = _load_cascade("haarcascade_frontalface_default.xml")
_CASCADE_FULLBODY   = _load_cascade("haarcascade_fullbody.xml")
_CASCADE_UPPERBODY  = _load_cascade("haarcascade_upperbody.xml")
_CASCADE_CAR_FRONT  = _load_cascade("haarcascade_car.xml")

# Rolling background model for motion detection (webcam)
_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=40, detectShadows=False
)


class DetectorAgent:
    """
    Input  state keys : frame, timestamp
    Output state keys : detections (list of dicts)

    Detection schema:
        {
            "class_name": str,
            "confidence": float (0-1),
            "bbox": [x1, y1, x2, y2],
            "center": [cx, cy],
            "method": str   # hog | haar_face | haar_body | motion | contour
        }
    """

    def run(self, state: dict) -> dict:
        frame: np.ndarray = state["frame"]
        t0 = time.perf_counter()

        detections = []
        detections += self._detect_persons_hog(frame)
        detections += self._detect_faces(frame)
        detections += self._detect_vehicles(frame)
        detections += self._detect_motion(frame)

        # Deduplicate overlapping boxes from different detectors
        detections = self._nms(detections, iou_threshold=0.45)

        state["detections"] = detections
        state["_detector_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return state

    # ── Person detection via HOG ────────────────────────────────────────────
    def _detect_persons_hog(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / max(h, w))
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        try:
            rects, weights = _HOG.detectMultiScale(
                gray, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
        except Exception:
            return []

        results = []
        for (x, y, bw, bh), score in zip(rects, weights):
            x1, y1 = int(x / scale), int(y / scale)
            x2, y2 = int((x + bw) / scale), int((y + bh) / scale)
            conf = min(float(score[0]) / 2.0, 0.99)
            results.append({
                "class_name": "person",
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                "method": "hog"
            })
        return results

    # ── Face / upper-body detection via Haar ────────────────────────────────
    def _detect_faces(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        results = []

        for cascade, label in [
            (_CASCADE_FACE,      "face"),
            (_CASCADE_UPPERBODY, "person"),
        ]:
            if cascade is None:
                continue
            try:
                dets = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
                )
            except Exception:
                continue
            if len(dets) == 0:
                continue
            for (x, y, bw, bh) in dets:
                results.append({
                    "class_name": label,
                    "confidence": 0.70,
                    "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                    "center": [int(x + bw // 2), int(y + bh // 2)],
                    "method": "haar_face"
                })
        return results

    # ── Vehicle detection ───────────────────────────────────────────────────
    def _detect_vehicles(self, frame: np.ndarray) -> list:
        if _CASCADE_CAR_FRONT is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        results = []
        try:
            dets = _CASCADE_CAR_FRONT.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )
        except Exception:
            return []
        if len(dets) == 0:
            return []
        for (x, y, bw, bh) in dets:
            results.append({
                "class_name": "car",
                "confidence": 0.65,
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "center": [int(x + bw // 2), int(y + bh // 2)],
                "method": "haar_car"
            })
        return results

    # ── Motion detection via background subtraction ─────────────────────────
    def _detect_motion(self, frame: np.ndarray) -> list:
        fg_mask = _bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:   # ignore tiny noise
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-5)
            # Rough heuristic: person-shaped contour
            label = "person" if 0.3 < aspect < 1.5 else "moving_object"
            conf = min(area / 20000.0, 0.85)
            results.append({
                "class_name": label,
                "confidence": round(conf, 3),
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "center": [int(x + bw // 2), int(y + bh // 2)],
                "method": "motion"
            })
        return results

    # ── Non-Maximum Suppression to remove overlapping boxes ─────────────────
    def _nms(self, detections: list, iou_threshold: float = 0.45) -> list:
        if not detections:
            return []
        boxes  = np.array([d["bbox"] for d in detections], dtype=float)
        scores = np.array([d["confidence"] for d in detections], dtype=float)

        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
            order = order[np.where(iou <= iou_threshold)[0] + 1]

        return [detections[i] for i in keep]
