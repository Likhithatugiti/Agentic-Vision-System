"""
Agentic Vision System
A LangGraph-powered multi-agent pipeline for real-time video event detection.
Author: Tugiti Likhitha
"""

import asyncio
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import json
import time

from agents.vision_graph import VisionGraph
from utils.event_store import EventStore

app = FastAPI(
    title="Agentic Vision System",
    description="LangGraph multi-agent pipeline for real-time video event detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

vision_graph = VisionGraph()
event_store = EventStore()


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.post("/analyze/frame")
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze a single uploaded image frame through the full agent pipeline."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    result = await vision_graph.process_frame(frame, source="upload")
    if result.get("alert"):
        event_store.add(result)

    return JSONResponse(result)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint — receives base64 frames, streams back agent decisions."""
    await websocket.accept()
    frame_count = 0
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                frame_b64 = msg["data"]
                img_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    frame_count += 1
                    result = await vision_graph.process_frame(
                        frame,
                        source="webcam",
                        frame_id=frame_count
                    )
                    if result.get("alert"):
                        event_store.add(result)

                    await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print(f"Client disconnected after {frame_count} frames")


@app.get("/events")
async def get_events(limit: int = 50):
    """Return recent triggered alert events."""
    return JSONResponse({"events": event_store.recent(limit)})


@app.delete("/events")
async def clear_events():
    event_store.clear()
    return JSONResponse({"status": "cleared"})


@app.get("/health")
async def health():
    return {"status": "ok", "agents": vision_graph.agent_names()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
