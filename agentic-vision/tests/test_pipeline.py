"""
Test suite for the Agentic Vision System.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.detector import DetectorAgent
from agents.analyser import AnalyserAgent
from agents.decider import DeciderAgent
from agents.logger_agent import LoggerAgent
from utils.event_store import EventStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def base_state(blank_frame):
    return {
        "frame": blank_frame,
        "frame_id": 1,
        "source": "test",
        "timestamp": time.time(),
        "detections": [],
        "scene_context": {},
        "decision": {},
        "alert": None,
    }


@pytest.fixture
def crowded_state(base_state):
    """State with 6 person detections — should trigger crowding rule."""
    base_state["detections"] = [
        {"class_name": "person", "confidence": 0.9,
         "bbox": [i*80, 100, i*80+60, 200], "center": [i*80+30, 150]}
        for i in range(6)
    ]
    return base_state


@pytest.fixture
def weapon_state(base_state):
    base_state["detections"] = [
        {"class_name": "knife", "confidence": 0.85,
         "bbox": [100, 100, 150, 200], "center": [125, 150]},
        {"class_name": "person", "confidence": 0.92,
         "bbox": [50, 50, 200, 300], "center": [125, 175]},
    ]
    return base_state


# ---------------------------------------------------------------------------
# DetectorAgent tests
# ---------------------------------------------------------------------------

class TestDetectorAgent:
    def test_returns_list(self, base_state):
        agent = DetectorAgent()
        result = agent.run(base_state)
        assert isinstance(result["detections"], list)

    def test_blank_frame_no_detections(self, base_state):
        """HOG on a blank frame should return zero or very few detections."""
        agent = DetectorAgent()
        result = agent.run(base_state)
        # blank frame should produce 0 detections regardless of backend
        assert len(result["detections"]) == 0

    def test_detection_schema(self, base_state):
        """If any detection is returned, it must have required keys."""
        agent = DetectorAgent()
        result = agent.run(base_state)
        for d in result["detections"]:
            assert "class_name" in d
            assert "confidence" in d
            assert "bbox" in d
            assert "center" in d
            assert 0.0 <= d["confidence"] <= 1.0
            assert len(d["bbox"]) == 4
            assert len(d["center"]) == 2

    def test_latency_key_added(self, base_state):
        agent = DetectorAgent()
        result = agent.run(base_state)
        assert "_detector_ms" in result


# ---------------------------------------------------------------------------
# AnalyserAgent tests
# ---------------------------------------------------------------------------

class TestAnalyserAgent:
    def test_scene_context_keys(self, crowded_state):
        crowded_state["detections"] = [
            {"class_name": "person", "confidence": 0.9,
             "bbox": [0, 0, 60, 100], "center": [30, 50]}
            for _ in range(3)
        ]
        agent = AnalyserAgent()
        result = agent.run(crowded_state)
        ctx = result["scene_context"]
        for key in ["person_count", "vehicle_count", "weapon_count",
                    "class_counts", "zones", "triggered_rules"]:
            assert key in ctx

    def test_crowding_rule_fires(self, crowded_state):
        agent = AnalyserAgent()
        result = agent.run(crowded_state)
        assert "crowding" in result["scene_context"]["triggered_rules"]

    def test_weapon_rule_fires(self, weapon_state):
        agent = AnalyserAgent()
        result = agent.run(weapon_state)
        assert "weapon_detected" in result["scene_context"]["triggered_rules"]

    def test_zone_distribution(self, base_state):
        base_state["detections"] = [
            {"class_name": "person", "confidence": 0.9,
             "bbox": [10, 10, 70, 100], "center": [40, 55]},   # left
            {"class_name": "person", "confidence": 0.9,
             "bbox": [290, 10, 350, 100], "center": [320, 55]}, # centre
            {"class_name": "person", "confidence": 0.9,
             "bbox": [560, 10, 620, 100], "center": [590, 55]}, # right
        ]
        agent = AnalyserAgent()
        result = agent.run(base_state)
        zones = result["scene_context"]["zones"]
        assert zones["left"] == 1
        assert zones["centre"] == 1
        assert zones["right"] == 1

    def test_empty_detections(self, base_state):
        agent = AnalyserAgent()
        result = agent.run(base_state)
        assert result["scene_context"]["person_count"] == 0
        assert result["scene_context"]["triggered_rules"] == []


# ---------------------------------------------------------------------------
# DeciderAgent tests
# ---------------------------------------------------------------------------

class TestDeciderAgent:
    def _run_pipeline(self, state):
        AnalyserAgent().run(state)
        DeciderAgent().run(state)
        return state

    def test_no_alert_on_empty(self, base_state):
        result = self._run_pipeline(base_state)
        assert result["decision"]["should_alert"] is False

    def test_critical_alert_on_weapon(self, weapon_state):
        result = self._run_pipeline(weapon_state)
        assert result["decision"]["should_alert"] is True
        assert result["decision"]["severity"] == "CRITICAL"

    def test_medium_alert_on_crowd(self, crowded_state):
        result = self._run_pipeline(crowded_state)
        assert result["decision"]["should_alert"] is True
        assert result["decision"]["severity"] in ("MEDIUM", "HIGH")

    def test_cooldown_suppression(self, weapon_state):
        """Firing the same rule twice rapidly should suppress the second."""
        agent = DeciderAgent()
        weapon_state["scene_context"] = {}
        AnalyserAgent().run(weapon_state)
        agent.run(weapon_state)
        first = weapon_state["decision"]["should_alert"]

        # Second call with same rules — cooldown not expired
        import copy
        state2 = copy.deepcopy(weapon_state)
        state2["decision"] = {}
        agent.run(state2)
        second = state2["decision"]["should_alert"]

        assert first is True
        assert second is False  # suppressed by cooldown


# ---------------------------------------------------------------------------
# LoggerAgent tests
# ---------------------------------------------------------------------------

class TestLoggerAgent:
    def _run(self, state, run_detector=True):
        if run_detector:
            DetectorAgent().run(state)
        AnalyserAgent().run(state)
        DeciderAgent().run(state)
        LoggerAgent().run(state)
        return state

    def test_alert_none_when_no_event(self, base_state):
        result = self._run(base_state)
        assert result["alert"] is None

    def test_alert_schema_on_weapon(self, weapon_state):
        # skip detector so mock detections (knife+person) are preserved
        result = self._run(weapon_state, run_detector=False)
        alert = result["alert"]
        assert alert is not None
        for key in ["alert_id", "timestamp", "severity", "primary_rule",
                    "description", "scene", "pipeline_ms"]:
            assert key in alert

    def test_pipeline_ms_positive(self, weapon_state):
        result = self._run(weapon_state, run_detector=False)
        if result["alert"]:
            assert result["alert"]["pipeline_ms"]["total"] >= 0


# ---------------------------------------------------------------------------
# EventStore tests
# ---------------------------------------------------------------------------

class TestEventStore:
    def test_add_and_recent(self):
        store = EventStore()
        store.add({"alert_id": "abc", "severity": "HIGH"})
        store.add({"alert_id": "def", "severity": "LOW"})
        recent = store.recent(10)
        assert len(recent) == 2
        assert recent[0]["alert_id"] == "def"  # newest first

    def test_clear(self):
        store = EventStore()
        store.add({"alert_id": "x"})
        store.clear()
        assert store.recent(10) == []

    def test_maxlen(self):
        store = EventStore(maxlen=3)
        for i in range(5):
            store.add({"alert_id": str(i)})
        assert len(store.recent(100)) == 3

    def test_stats(self):
        store = EventStore()
        store.add({"severity": "HIGH",     "primary_rule": "crowding"})
        store.add({"severity": "CRITICAL", "primary_rule": "weapon_detected"})
        store.add({"severity": "HIGH",     "primary_rule": "high_density"})
        stats = store.stats()
        assert stats["total_alerts"] == 3
        assert stats["by_severity"]["HIGH"] == 2
