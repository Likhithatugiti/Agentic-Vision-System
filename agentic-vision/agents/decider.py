"""
DeciderAgent — Node 3 in the VisionGraph.

Core decision logic: given scene_context, should an alert be triggered?
If yes, what severity (LOW / MEDIUM / HIGH / CRITICAL)?

Decision policy:
  CRITICAL  — weapon detected
  HIGH      — high density crowd (≥10 people) OR multi-class complex scene
  MEDIUM    — general crowding (≥5 people) OR vehicle present with people
  LOW       — lone person in frame (surveillance ping)
  NONE      — no significant event

Also handles alert suppression (don't re-fire same rule within cooldown window).
"""

import time
from collections import defaultdict


# Seconds before the same rule can fire again
COOLDOWN = {
    "weapon_detected":  5,
    "high_density":    10,
    "crowding":        15,
    "vehicle_present": 20,
    "lone_person":     30,
    "multi_class":     10,
}

# Rule → severity mapping
SEVERITY_MAP = {
    "weapon_detected": "CRITICAL",
    "high_density":    "HIGH",
    "multi_class":     "HIGH",
    "crowding":        "MEDIUM",
    "vehicle_present": "MEDIUM",
    "lone_person":     "LOW",
}

# Human-readable descriptions
DESCRIPTIONS = {
    "weapon_detected": "Potential weapon detected in frame.",
    "high_density":    "High-density crowd detected (≥10 persons).",
    "multi_class":     "Complex multi-class scene — multiple object types present.",
    "crowding":        "Crowd threshold exceeded (≥5 persons).",
    "vehicle_present": "Vehicle(s) detected alongside persons.",
    "lone_person":     "Single person detected in monitored zone.",
}

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class DeciderAgent:
    """
    Input  state keys : scene_context
    Output state keys : decision
    """

    def __init__(self):
        self._last_fired: dict[str, float] = defaultdict(float)

    def run(self, state: dict) -> dict:
        t0 = time.perf_counter()
        ctx = state.get("scene_context", {})
        triggered = ctx.get("triggered_rules", [])
        now = time.time()

        # Filter out rules still in cooldown
        active_rules = [
            rule for rule in triggered
            if now - self._last_fired[rule] > COOLDOWN.get(rule, 15)
        ]

        if not active_rules:
            state["decision"] = {
                "should_alert": False,
                "reason": "no_active_rules",
                "severity": None,
                "active_rules": [],
                "suppressed_rules": [r for r in triggered if r not in active_rules],
            }
            state["_decider_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return state

        # Pick highest severity among active rules
        best_rule = None
        best_severity = None
        for sev in SEVERITY_ORDER:
            for rule in active_rules:
                if SEVERITY_MAP.get(rule) == sev:
                    best_rule = rule
                    best_severity = sev
                    break
            if best_rule:
                break

        # Update cooldown timestamp
        for rule in active_rules:
            self._last_fired[rule] = now

        state["decision"] = {
            "should_alert":  True,
            "primary_rule":  best_rule,
            "severity":      best_severity,
            "description":   DESCRIPTIONS.get(best_rule, "Event detected."),
            "active_rules":  active_rules,
            "suppressed_rules": [r for r in triggered if r not in active_rules],
        }
        state["_decider_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return state
