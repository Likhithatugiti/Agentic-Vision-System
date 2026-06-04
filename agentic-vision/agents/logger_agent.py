"""
LoggerAgent — Node 4 (final) in the VisionGraph.

Assembles the final structured alert payload from all upstream state.
If decision.should_alert is False, alert remains None.

Output alert schema:
{
    "alert_id":     str,          # unique ID for dedup / tracing
    "timestamp":    float,        # unix epoch
    "source":       str,          # webcam | upload | test
    "frame_id":     int,
    "severity":     str,          # LOW | MEDIUM | HIGH | CRITICAL
    "primary_rule": str,
    "description":  str,
    "scene": {
        "person_count":  int,
        "vehicle_count": int,
        "weapon_count":  int,
        "total_objects": int,
        "class_counts":  dict,
        "zones":         dict,
    },
    "pipeline_ms": {              # per-agent latency breakdown
        "detector":  float,
        "analyser":  float,
        "decider":   float,
        "total":     float,
    }
}
"""

import time
import uuid


class LoggerAgent:
    """
    Input  state keys : decision, scene_context, frame_id, source, timestamp
    Output state keys : alert (dict | None)
    """

    def run(self, state: dict) -> dict:
        t0 = time.perf_counter()
        decision = state.get("decision", {})

        if not decision.get("should_alert"):
            state["alert"] = None
            return state

        ctx = state.get("scene_context", {})
        total_ms = (
            state.get("_detector_ms", 0)
            + state.get("_analyser_ms", 0)
            + state.get("_decider_ms", 0)
        )

        alert = {
            "alert_id":     str(uuid.uuid4())[:8],
            "timestamp":    state.get("timestamp", time.time()),
            "source":       state.get("source", "unknown"),
            "frame_id":     state.get("frame_id", 0),
            "severity":     decision.get("severity"),
            "primary_rule": decision.get("primary_rule"),
            "description":  decision.get("description", ""),
            "active_rules": decision.get("active_rules", []),
            "scene": {
                "person_count":  ctx.get("person_count", 0),
                "vehicle_count": ctx.get("vehicle_count", 0),
                "weapon_count":  ctx.get("weapon_count", 0),
                "total_objects": ctx.get("total_objects", 0),
                "class_counts":  ctx.get("class_counts", {}),
                "zones":         ctx.get("zones", {}),
                "avg_confidence": ctx.get("avg_confidence", 0),
            },
            "pipeline_ms": {
                "detector": state.get("_detector_ms", 0),
                "analyser":  state.get("_analyser_ms", 0),
                "decider":   state.get("_decider_ms", 0),
                "total":     round(total_ms + (time.perf_counter() - t0) * 1000, 1),
            }
        }

        state["alert"] = alert
        return state
