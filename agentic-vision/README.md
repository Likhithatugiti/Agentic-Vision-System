# Agentic Vision System

A **LangGraph-powered multi-agent pipeline** for real-time video event detection. Combines YOLOv8 computer vision with a directed agent graph to detect, analyse, and act on visual events — with a production FastAPI backend and live WebSocket streaming.

Built by **Tugiti Likhitha** | IIT Kharagpur

---

## Architecture

```
Video Frame
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                  │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────┐ │
│  │ Detector │──▶│ Analyser │──▶│ Decider  │──▶│Logger│ │
│  │  Agent   │   │  Agent   │   │  Agent   │   │Agent │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┘ │
│  YOLOv8 inference  Scene context  Alert decision  Payload│
└─────────────────────────────────────────────────────────┘
    │
    ▼
FastAPI REST + WebSocket  ──▶  Live Dashboard UI
```

### Agents

| Agent | Role | Output |
|---|---|---|
| **DetectorAgent** | Runs YOLOv8n inference; filters to 11 tracked classes | `detections[]` with bbox, class, confidence |
| **AnalyserAgent** | Builds scene context: crowd density, zone mapping, rule evaluation | `scene_context` with `triggered_rules[]` |
| **DeciderAgent** | Applies severity policy + cooldown suppression to decide alert | `decision` with severity (LOW/MEDIUM/HIGH/CRITICAL) |
| **LoggerAgent** | Formats structured alert payload with per-agent latency breakdown | `alert` dict |

### Event Rules

| Rule | Condition | Severity |
|---|---|---|
| `weapon_detected` | knife/scissors in frame | CRITICAL |
| `high_density` | ≥10 persons | HIGH |
| `multi_class` | ≥3 distinct object classes | HIGH |
| `crowding` | ≥5 persons | MEDIUM |
| `vehicle_present` | vehicle + person | MEDIUM |
| `lone_person` | exactly 1 person | LOW |

---

## Setup

```bash
# Clone and install
git clone <repo>
cd agentic-vision
pip install -r requirements.txt

# Run (YOLOv8n weights download automatically on first run ~6MB)
python main.py
# Open http://localhost:8000
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `POST` | `/analyze/frame` | Analyse a single uploaded image |
| `WS` | `/ws/stream` | Real-time WebSocket frame stream |
| `GET` | `/events` | Fetch recent alert log |
| `DELETE` | `/events` | Clear alert log |
| `GET` | `/health` | Health check + agent names |

### WebSocket protocol

**Send** (client → server):
```json
{ "type": "frame", "data": "<base64-jpeg>" }
```

**Receive** (server → client):
```json
{
  "frame_id": 42,
  "detections": [{ "class_name": "person", "confidence": 0.91, "bbox": [x1,y1,x2,y2] }],
  "scene_context": { "person_count": 2, "triggered_rules": ["crowding"] },
  "decision": { "should_alert": true, "severity": "MEDIUM" },
  "alert": {
    "alert_id": "a3f2b1c0",
    "severity": "MEDIUM",
    "description": "Crowd threshold exceeded (≥5 persons).",
    "scene": { "person_count": 6, "total_objects": 6 },
    "pipeline_ms": { "detector": 18.2, "analyser": 0.4, "decider": 0.1, "total": 19.1 }
  }
}
```

---

## Tests

```bash
pytest tests/ -v
```

17 tests covering:
- DetectorAgent: schema validation, blank-frame handling, latency keys
- AnalyserAgent: rule firing, zone distribution, empty detection handling
- DeciderAgent: severity mapping, cooldown suppression
- LoggerAgent: alert schema, null alert on no event
- EventStore: FIFO ordering, maxlen, stats aggregation

---

## Tech Stack

- **YOLOv8n** (Ultralytics) — real-time object detection
- **LangGraph** — directed agent StateGraph orchestration
- **FastAPI** — async REST + WebSocket server
- **OpenCV** — frame processing + HOG fallback detector
- **pytest** — full test suite with fixtures

---

## Design Decisions

**Why LangGraph over a simple function chain?**
The StateGraph gives explicit node boundaries, making it trivial to swap agents (e.g. replace DetectorAgent with a different model), add parallel branches, or inject human-in-the-loop checkpoints — exactly what a production agentic system needs.

**Why a cooldown system in DeciderAgent?**
Without cooldown suppression, a crowded scene fires an alert every frame. Cooldowns make the alert stream semantically meaningful: each alert represents a *new* event occurrence, not a polling tick.

**Graceful degradation**
If `ultralytics` is not installed, DetectorAgent falls back to OpenCV HOG. If `langgraph` is not installed, the pipeline runs sequentially. The system is always runnable.
