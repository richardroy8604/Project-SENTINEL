"""
ULTRON Monitor Station — Decoupled IoT Web App & MQTT Hub
=========================================================
A completely independent remote monitoring station.
Connects to HiveMQ Cloud over TLS (Port 8883), consumes ULTRON's telemetry,
chat logs, forensic snapshots, and alerts, and serves the Tactical PWA
Web Dashboard over local network WebSockets.
"""

import asyncio
import json
import os
import socket
import ssl
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
except ImportError:
    print("[MONITOR] Error: paho-mqtt required. Run: pip install paho-mqtt")
    sys.exit(1)

# Paths
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "received_snapshots"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Auto-load .env
_env_files = [BASE_DIR / ".env", BASE_DIR.parent / ".env"]
for _ef in _env_files:
    if _ef.exists():
        with open(_ef, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v

# Broker Credentials
BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "2fd0cc9fccec48b9befe7b3d4b75cc59.s1.eu.hivemq.cloud")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", 8883))
USERNAME = os.getenv("MQTT_USERNAME", "hivemq.webclient.1790576042940")
PASSWORD = os.getenv("MQTT_PASSWORD", "nMJJC!GVDRqO*cKXz%Rhcyiwz7c6FPCq")
TOPIC_SUBSCRIPTION = "ultron/#"
PORT = int(os.getenv("MONITOR_PORT", 8000))


def get_local_ip() -> str:
    """Detect LAN IPv4 address for remote mobile phone connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# State Cache
class MonitorState:
    def __init__(self):
        self.lock = threading.Lock()
        self.security_state = "PATROL"
        self.person_count = 0
        self.persons = []
        self.chat_history = []
        self.latest_snapshot = None
        self.recent_snapshots = []
        self.mqtt_connected = False
        self.last_heartbeat = time.time()


state = MonitorState()
active_websockets: Set[WebSocket] = set()
event_loop: Optional[asyncio.AbstractEventLoop] = None

app = FastAPI(title="ULTRON Remote Monitor", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def broadcast_to_clients(payload: dict):
    """Thread-safe WebSocket broadcaster."""
    global event_loop, active_websockets
    if not active_websockets or not event_loop:
        return

    msg_str = json.dumps(payload)

    async def _send():
        dead = set()
        for ws in list(active_websockets):
            try:
                await ws.send_text(msg_str)
            except Exception:
                dead.add(ws)
        active_websockets.difference_update(dead)

    try:
        asyncio.run_coroutine_threadsafe(_send(), event_loop)
    except Exception:
        pass


# ── MQTT Client Setup ───────────────────────────────────────────────
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0 or str(rc) == "Success":
        state.mqtt_connected = True
        print(f"[MONITOR MQTT] Connected to HiveMQ Cloud ({BROKER_HOST}:{BROKER_PORT})")
        print(f"[MONITOR MQTT] Subscribed to {TOPIC_SUBSCRIPTION}")
        client.subscribe(TOPIC_SUBSCRIPTION, qos=1)
        broadcast_to_clients({
            "type": "init",
            "data": {
                "security_state": state.security_state,
                "person_count": state.person_count,
                "persons": state.persons,
                "chat_history": state.chat_history[-20:],
                "latest_snapshot": state.latest_snapshot,
                "recent_snapshots": state.recent_snapshots[:12],
                "mqtt_connected": True,
                "mqtt_broker": BROKER_HOST,
                "timestamp": time.time(),
            },
        })
    else:
        state.mqtt_connected = False
        print(f"[MONITOR MQTT] Connection failed: {rc}")


def on_message(client, userdata, msg):
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return

    if "state" in topic:
        with state.lock:
            state.security_state = payload.get("security_state", state.security_state)
            state.person_count = payload.get("person_count", state.person_count)
            state.persons = payload.get("persons", state.persons)
        broadcast_to_clients({
            "type": "state_changed",
            "data": payload,
        })

    elif "chat" in topic:
        with state.lock:
            state.chat_history.append(payload)
            if len(state.chat_history) > 40:
                state.chat_history = state.chat_history[-40:]
        broadcast_to_clients({
            "type": "chat_message",
            "data": payload,
        })

    elif "snapshot" in topic:
        with state.lock:
            state.latest_snapshot = payload
            state.recent_snapshots.insert(0, payload)
            if len(state.recent_snapshots) > 25:
                state.recent_snapshots = state.recent_snapshots[:25]

        broadcast_to_clients({
            "type": "snapshot_captured",
            "data": payload,
        })

    elif "alerts" in topic:
        broadcast_to_clients({
            "type": "alert",
            "data": payload,
        })

    elif "heartbeat" in topic:
        state.last_heartbeat = time.time()


# ── REST & WebSocket Routes ─────────────────────────────────────────
@app.get("/")
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"status": "ULTRON Monitor Online"})


@app.get("/manifest.json")
async def get_manifest():
    manifest_file = STATIC_DIR / "manifest.json"
    if manifest_file.exists():
        return FileResponse(manifest_file)
    return JSONResponse({"name": "ULTRON Security Companion"})


@app.get("/api/status")
async def get_status():
    with state.lock:
        return {
            "security_state": state.security_state,
            "person_count": state.person_count,
            "persons": state.persons,
            "mqtt_connected": state.mqtt_connected,
            "mqtt_broker": BROKER_HOST,
            "timestamp": time.time(),
        }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        # Send initial state snapshot
        with state.lock:
            init_payload = {
                "type": "init",
                "data": {
                    "security_state": state.security_state,
                    "person_count": state.person_count,
                    "persons": state.persons,
                    "chat_history": state.chat_history[-20:],
                    "latest_snapshot": state.latest_snapshot,
                    "recent_snapshots": state.recent_snapshots[:12],
                    "mqtt_connected": state.mqtt_connected,
                    "mqtt_broker": BROKER_HOST,
                    "timestamp": time.time(),
                },
            }
        await websocket.send_text(json.dumps(init_payload))

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        active_websockets.discard(websocket)
    except Exception:
        active_websockets.discard(websocket)


def start_mqtt_client():
    client_id = f"ultron_monitor_hub_{int(time.time())}"
    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id=client_id)
    client.tls_set()
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect_async(BROKER_HOST, BROKER_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"[MONITOR MQTT] Connection error: {e}")


def main():
    global event_loop
    print("=" * 65)
    print("  U L T R O N  M O N I T O R  S T A T I O N")
    print("=" * 65)
    print(f"Connecting to HiveMQ Cloud: {BROKER_HOST}:{BROKER_PORT}...")
    start_mqtt_client()

    local_ip = get_local_ip()
    print()
    print("Remote Monitor Dashboard ready:")
    print(f"  Local Browser:  http://localhost:{PORT}")
    print(f"  Mobile Device:  http://{local_ip}:{PORT}  (Open on your iPhone)")
    print("=" * 65)

    uvicorn_cfg = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uvicorn_cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    event_loop = loop
    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    main()
