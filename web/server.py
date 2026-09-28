"""
ULTRON Web — Remote Companion Server & WebSocket Hub
====================================================
Serves the mobile-optimized Tactical PWA Dashboard and broadcasts real-time
system events, live chat dialogue, incident snapshots, and security telemetry.
"""

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import config
from core.event_bus import EventBus, EventTypes, Event
from core.context import ContextManager
from security.snapshot_manager import SnapshotManager


def get_local_ip() -> str:
    """Detect LAN IPv4 address for remote mobile/laptop connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class WebServer:
    """
    FastAPI + Uvicorn server running in a background thread.
    Bridges EventBus events directly into connected WebSocket clients.
    """

    def __init__(
        self,
        event_bus: EventBus,
        context: ContextManager,
        snapshot_manager: SnapshotManager,
        mqtt_dispatcher=None,
    ):
        self.event_bus = event_bus
        self.context = context
        self.snapshot_manager = snapshot_manager
        self.mqtt_dispatcher = mqtt_dispatcher

        self.host = getattr(config, "WEB_SERVER_HOST", "0.0.0.0")
        self.port = getattr(config, "WEB_SERVER_PORT", 8000)
        self.enabled = getattr(config, "WEB_SERVER_ENABLED", True)

        self._active_connections: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._running = False

        self.app = FastAPI(title="ULTRON Remote Companion", version="1.0.0")
        self._setup_app()
        self._subscribe_events()

    def _setup_app(self):
        """Configure FastAPI routes, CORS, and static file mounts."""
        app = self.app

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir = Path(config.SNAPSHOTS_DIR)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        app.mount("/api/snapshots", StaticFiles(directory=str(snapshots_dir)), name="snapshots")
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def get_index():
            index_file = static_dir / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return JSONResponse({"status": "ULTRON Companion API Online"})

        @app.get("/manifest.json")
        async def get_manifest():
            manifest_file = static_dir / "manifest.json"
            if manifest_file.exists():
                return FileResponse(manifest_file)
            return JSONResponse({"name": "ULTRON Security Companion"})

        @app.get("/api/status")
        async def get_status():
            mqtt_connected = self.mqtt_dispatcher.is_connected if self.mqtt_dispatcher else False
            return {
                "security_state": self.context._security_state,
                "person_count": self.context.person_count,
                "persons": [
                    {
                        "id": p.track_id,
                        "dwell_s": round(p.dwell_time, 1),
                        "phone": p.holding_phone,
                        "weapon": p.weapon_type if p.holding_weapon else None,
                    }
                    for p in self.context.persons
                ],
                "mqtt_connected": mqtt_connected,
                "mqtt_broker": getattr(config, "MQTT_BROKER_HOST", "broker.hivemq.com"),
                "timestamp": time.time(),
            }

        @app.get("/api/history/chat")
        async def get_chat_history():
            return {
                "messages": self.context.conversation_history[-30:]
            }

        @app.get("/api/history/snapshots")
        async def get_snapshots_history(limit: int = 20):
            return {
                "snapshots": self.snapshot_manager.get_recent_snapshots(limit=limit)
            }

        @app.get("/api/snapshot/latest")
        async def get_latest_snapshot():
            latest = self.snapshot_manager.get_latest_snapshot()
            if latest and latest.get("filepath") and os.path.exists(latest["filepath"]):
                return FileResponse(latest["filepath"], media_type="image/jpeg")
            return JSONResponse({"error": "No snapshots recorded yet"}, status_code=404)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._active_connections.add(websocket)
            try:
                # Send initial state synchronization handshake
                mqtt_connected = self.mqtt_dispatcher.is_connected if self.mqtt_dispatcher else False
                latest_snap = self.snapshot_manager.get_latest_snapshot()
                init_payload = {
                    "type": "init",
                    "data": {
                        "security_state": self.context._security_state,
                        "person_count": self.context.person_count,
                        "persons": [
                            {
                                "id": p.track_id,
                                "dwell_s": round(p.dwell_time, 1),
                                "phone": p.holding_phone,
                                "weapon": p.weapon_type if p.holding_weapon else None,
                            }
                            for p in self.context.persons
                        ],
                        "chat_history": self.context.conversation_history[-20:],
                        "latest_snapshot": latest_snap,
                        "recent_snapshots": self.snapshot_manager.get_recent_snapshots(limit=12),
                        "mqtt_connected": mqtt_connected,
                        "mqtt_broker": getattr(config, "MQTT_BROKER_HOST", "broker.hivemq.com"),
                        "timestamp": time.time(),
                    },
                }
                await websocket.send_text(json.dumps(init_payload))

                # Keep connection alive listening for ping
                while True:
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))

            except WebSocketDisconnect:
                self._active_connections.discard(websocket)
            except Exception:
                self._active_connections.discard(websocket)

    def _subscribe_events(self):
        """Wire EventBus events to WebSocket broadcast."""
        self.event_bus.subscribe(EventTypes.STATE_CHANGED, self._broadcast_state_changed)
        self.event_bus.subscribe(EventTypes.SPEECH_RECOGNIZED, self._broadcast_speech_recognized)
        self.event_bus.subscribe(EventTypes.RESPONSE_GENERATED, self._broadcast_response_generated)
        self.event_bus.subscribe(EventTypes.SNAPSHOT_CAPTURED, self._broadcast_snapshot_captured)
        self.event_bus.subscribe(EventTypes.WEAPON_DETECTED, self._broadcast_weapon_detected)
        self.event_bus.subscribe(EventTypes.CAMERA_OBSTRUCTED, self._broadcast_camera_obstructed)
        self.event_bus.subscribe(EventTypes.LOITERING_DETECTED, self._broadcast_loitering_detected)
        self.event_bus.subscribe(EventTypes.SPEAKING_STARTED, self._broadcast_speaking_started)
        self.event_bus.subscribe(EventTypes.SPEAKING_FINISHED, self._broadcast_speaking_finished)

    def _broadcast_json(self, payload: dict):
        """Thread-safe WebSocket broadcaster."""
        if not self._active_connections or not self._loop or not self._running:
            return

        msg_str = json.dumps(payload)

        async def _send():
            dead_connections = set()
            for ws in list(self._active_connections):
                try:
                    await ws.send_text(msg_str)
                except Exception:
                    dead_connections.add(ws)
            self._active_connections -= dead_connections

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop)
        except Exception:
            pass

    # ── Event Bus Callbacks ─────────────────────────────────────────

    def _broadcast_state_changed(self, event: Event):
        self._broadcast_json({
            "type": "state_changed",
            "data": {
                "security_state": event.data.get("new_state", "UNKNOWN"),
                "old_state": event.data.get("old_state", "UNKNOWN"),
                "reason": event.data.get("reason", ""),
                "person_count": self.context.person_count,
                "timestamp": event.timestamp,
            },
        })

    def _broadcast_speech_recognized(self, event: Event):
        self._broadcast_json({
            "type": "chat_message",
            "data": {
                "role": "visitor",
                "text": event.data.get("text", ""),
                "latency_ms": event.data.get("latency_ms", 0.0),
                "timestamp": event.timestamp,
            },
        })

    def _broadcast_response_generated(self, event: Event):
        self._broadcast_json({
            "type": "chat_message",
            "data": {
                "role": "ultron",
                "text": event.data.get("text", ""),
                "latency_ms": event.data.get("latency_ms", 0.0),
                "autonomous": event.data.get("autonomous", False),
                "timestamp": event.timestamp,
            },
        })

    def _broadcast_snapshot_captured(self, event: Event):
        self._broadcast_json({
            "type": "snapshot_captured",
            "data": event.data,
        })

    def _broadcast_weapon_detected(self, event: Event):
        self._broadcast_json({
            "type": "alert",
            "data": {
                "alert_type": "ARMED_THREAT",
                "severity": "CRITICAL",
                "track_id": event.data.get("track_id", -1),
                "weapon": event.data.get("weapon", "WEAPON"),
                "timestamp": event.timestamp,
            },
        })

    def _broadcast_camera_obstructed(self, event: Event):
        self._broadcast_json({
            "type": "alert",
            "data": {
                "alert_type": "CAMERA_TAMPER",
                "severity": "CRITICAL",
                "type": event.data.get("type", "LENS_COVERED"),
                "timestamp": event.timestamp,
            },
        })

    def _broadcast_loitering_detected(self, event: Event):
        self._broadcast_json({
            "type": "alert",
            "data": {
                "alert_type": "LOITERING",
                "severity": "WARNING",
                "track_id": event.data.get("track_id", -1),
                "duration_s": event.data.get("duration", 90.0),
                "timestamp": event.timestamp,
            },
        })

    def _broadcast_speaking_started(self, event: Event):
        self._broadcast_json({
            "type": "speaking_status",
            "data": {"is_speaking": True, "text": event.data.get("text", "")},
        })

    def _broadcast_speaking_finished(self, event: Event):
        self._broadcast_json({
            "type": "speaking_status",
            "data": {"is_speaking": False},
        })

    # ── Server Lifecycle ────────────────────────────────────────────

    def start(self) -> bool:
        """Start the Uvicorn web server in a daemon thread."""
        if not self.enabled:
            return False

        if self._running:
            return True

        self._running = True
        local_ip = get_local_ip()

        def _run_server():
            # Create a dedicated event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            uvicorn_cfg = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                loop="asyncio",
                log_level="warning",
                access_log=False,
            )
            self._uvicorn_server = uvicorn.Server(uvicorn_cfg)
            print(f"[ULTRON Web] Companion Dashboard running:")
            print(f"             Local:   http://localhost:{self.port}")
            print(f"             Network: http://{local_ip}:{self.port}  (Open this on your iPhone/laptop)")
            loop.run_until_complete(self._uvicorn_server.serve())

        self._server_thread = threading.Thread(
            target=_run_server,
            name="UltronWebServer",
            daemon=True,
        )
        self._server_thread.start()
        return True

    def stop(self):
        """Stop the web server and close connections."""
        self._running = False
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        print("[ULTRON Web] Web server stopped.")
