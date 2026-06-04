"""
VisionGraph — LangGraph-based multi-agent pipeline.

Nodes:
  1. DetectorAgent    — runs YOLOv8 on the frame, extracts detections
  2. AnalyserAgent    — classifies scene context, checks threat/event rules
  3. DeciderAgent     — decides whether to trigger an alert and its severity
  4. LoggerAgent      — formats the structured alert payload

State flows left-to-right through a compiled StateGraph.
"""

import asyncio
import time
from typing import TypedDict, List, Optional, Any
import cv2
import numpy as np

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from agents.detector import DetectorAgent
from agents.analyser import AnalyserAgent
from agents.decider import DeciderAgent
from agents.logger_agent import LoggerAgent


class VisionState(TypedDict):
    frame: Any                        # raw numpy frame
    frame_id: int
    source: str
    timestamp: float
    detections: List[dict]            # from DetectorAgent
    scene_context: dict               # from AnalyserAgent
    decision: dict                    # from DeciderAgent
    alert: Optional[dict]             # final payload from LoggerAgent


class VisionGraph:
    def __init__(self):
        self.detector = DetectorAgent()
        self.analyser = AnalyserAgent()
        self.decider = DeciderAgent()
        self.logger = LoggerAgent()

        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
        else:
            self.graph = None

    def _build_graph(self):
        graph = StateGraph(VisionState)

        graph.add_node("detector", self.detector.run)
        graph.add_node("analyser", self.analyser.run)
        graph.add_node("decider", self.decider.run)
        graph.add_node("logger", self.logger.run)

        graph.set_entry_point("detector")
        graph.add_edge("detector", "analyser")
        graph.add_edge("analyser", "decider")
        graph.add_edge("decider", "logger")
        graph.add_edge("logger", END)

        return graph.compile()

    async def process_frame(self, frame: np.ndarray, source: str = "unknown", frame_id: int = 0) -> dict:
        initial_state: VisionState = {
            "frame": frame,
            "frame_id": frame_id,
            "source": source,
            "timestamp": time.time(),
            "detections": [],
            "scene_context": {},
            "decision": {},
            "alert": None,
        }

        if self.graph:
            # LangGraph compiled execution
            final_state = await asyncio.get_event_loop().run_in_executor(
                None, self.graph.invoke, initial_state
            )
        else:
            # Fallback: sequential execution without LangGraph
            state = initial_state
            state = self.detector.run(state)
            state = self.analyser.run(state)
            state = self.decider.run(state)
            state = self.logger.run(state)
            final_state = state

        return self._serialize(final_state)

    def _serialize(self, state: dict) -> dict:
        """Strip numpy arrays before JSON serialisation."""
        return {
            "frame_id": state.get("frame_id"),
            "source": state.get("source"),
            "timestamp": state.get("timestamp"),
            "detections": state.get("detections", []),
            "scene_context": state.get("scene_context", {}),
            "decision": state.get("decision", {}),
            "alert": state.get("alert"),
        }

    def agent_names(self):
        return ["DetectorAgent", "AnalyserAgent", "DeciderAgent", "LoggerAgent"]
