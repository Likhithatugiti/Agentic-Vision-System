"""
AnalyserAgent — Node 2 in the VisionGraph.

Takes the raw detections from DetectorAgent and builds semantic scene context:
  - crowd density estimate
  - detected object category summary
  - zone tagging (frame thirds: left / centre / right)
  - active rule flags (which event rules are potentially triggered)

No external model calls — pure rule-based analysis over detection results.
This keeps latency low and the pipeline deterministic.
"""

import time
from collections import Counter
from typing import Dict


# ---------------------------------------------------------------------------
# Event rule definitions
# Each rule maps to a condition over the scene context.
# DeciderAgent uses these flags to make its alert decision.
# ---------------------------------------------------------------------------
EVENT_RULES = {
    "crowding":         lambda ctx: ctx["person_count"] >= 5,
    "lone_person":      lambda ctx: ctx["person_count"] == 1,
    "vehicle_present":  lambda ctx: ctx["vehicle_count"] >= 1,
    "high_density":     lambda ctx: ctx["person_count"] >= 10,
    "weapon_detected":  lambda ctx: ctx["weapon_count"] >= 1,
    "multi_class":      lambda ctx: len(ctx["class_counts"]) >= 3,
}


class AnalyserAgent:
    """
    Input  state keys : detections
    Output state keys : scene_context
    """

    def run(self, state: dict) -> dict:
        t0 = time.perf_counter()
        detections = state.get("detections", [])
        frame = state.get("frame")

        h, w = (frame.shape[:2] if frame is not None else (480, 640))
        class_counts = Counter(d["class_name"] for d in detections)

        person_count  = class_counts.get("person", 0)
        vehicle_count = sum(class_counts.get(c, 0)
                            for c in ["car", "truck", "bus", "motorcycle", "bicycle"])
        weapon_count  = sum(class_counts.get(c, 0) for c in ["knife", "scissors"])

        # Zone tagging: which horizontal third is each person in?
        zones = {"left": 0, "centre": 0, "right": 0}
        for d in detections:
            if d["class_name"] == "person":
                cx = d["center"][0]
                if cx < w / 3:
                    zones["left"] += 1
                elif cx < 2 * w / 3:
                    zones["centre"] += 1
                else:
                    zones["right"] += 1

        avg_confidence = (
            round(sum(d["confidence"] for d in detections) / len(detections), 3)
            if detections else 0.0
        )

        scene_context = {
            "person_count":    person_count,
            "vehicle_count":   vehicle_count,
            "weapon_count":    weapon_count,
            "total_objects":   len(detections),
            "class_counts":    dict(class_counts),
            "zones":           zones,
            "avg_confidence":  avg_confidence,
            "frame_width":     w,
            "frame_height":    h,
        }

        # Evaluate event rules
        triggered_rules = [
            rule for rule, condition in EVENT_RULES.items()
            if condition(scene_context)
        ]
        scene_context["triggered_rules"] = triggered_rules

        state["scene_context"] = scene_context
        state["_analyser_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return state
