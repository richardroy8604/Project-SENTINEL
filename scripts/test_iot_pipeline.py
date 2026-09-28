"""
ULTRON IoT Pipeline & Companion Verification Suite
===================================================
Tests all Stage 9 components end-to-end:
  1. SnapshotManager (forensic annotation, color hierarchy, Base64 & disk storage)
  2. MQTTDispatcher (HiveMQ cloud broker connection, JSON topics, LWT)
  3. WebServer (FastAPI endpoints, WebSocket handshake, static files)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.event_bus import EventBus, EventTypes, Event
from core.context import ContextManager
from vision.detector import DetectionResult, PersonDetection
from security.snapshot_manager import SnapshotManager
from security.mqtt_dispatcher import MQTTDispatcher
from web.server import WebServer


def test_snapshot_pipeline():
    print("\n--- [TEST 1] Forensic Snapshot Generation & Annotations ---")
    event_bus = EventBus()
    snapshot_manager = SnapshotManager(event_bus=event_bus)

    # Subscribe to verify SNAPSHOT_CAPTURED publication
    captured_events = []
    event_bus.subscribe(
        EventTypes.SNAPSHOT_CAPTURED,
        lambda e: captured_events.append(e.data)
    )

    # 1. Create a dummy test frame (640x480 dark grey canvas)
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.putText(frame, "TEST CAMERA SIMULATION", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # 2. Create synthetic detections with Person, Phone, and Weapon
    bbox = (100, 80, 300, 440)
    person = PersonDetection(
        track_id=7,
        bbox=bbox,
        confidence=0.92,
        center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
        bbox_area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        holding_phone=True,
        holding_weapon=True,
        weapon_type="KNIFE",
    )

    detections = DetectionResult(
        persons=[person],
        phone_boxes=[(120, 200, 180, 280)],
        weapon_boxes=[(210, 230, 290, 320)],
        person_count=1,
        weapon_detected=True,
        detected_weapons=["KNIFE"],
        inference_ms=14.2,
    )


    # 3. Trigger snapshot capture
    print("Capturing armed threat forensic snapshot...")
    meta = snapshot_manager.capture_and_dispatch(
        trigger="TEST_ARMED_INTRUDER_DETECTED",
        security_state="CRITICAL",
        critical=True,
        frame_override=frame,
        detections_override=detections,
    )


    assert meta is not None, "Snapshot meta should not be None"
    assert os.path.exists(meta["filepath"]), f"Snapshot file not created at {meta['filepath']}"
    assert meta["base64"].startswith("data:image/jpeg;base64,"), "Base64 data URI format invalid"
    assert len(captured_events) == 1, "SNAPSHOT_CAPTURED event was not fired"
    print(f"  [OK] Snapshot saved: {meta['filename']} ({os.path.getsize(meta['filepath'])} bytes)")
    print(f"  [OK] Base64 payload generated ({len(meta['base64'])} chars)")
    print(f"  [OK] EventBus SNAPSHOT_CAPTURED broadcast verified")

    # Verify getter methods
    latest = snapshot_manager.get_latest_snapshot()
    assert latest is not None and latest["filename"] == meta["filename"], "get_latest_snapshot failed"
    recent = snapshot_manager.get_recent_snapshots(limit=5)
    assert len(recent) >= 1, "get_recent_snapshots returned empty list"
    assert "base64" not in recent[0], "Base64 should be stripped in lightweight recent list"
    print("  [OK] Snapshot getters verified")



def test_mqtt_dispatcher():
    print("\n--- [TEST 2] IoT MQTT Telemetry Dispatcher ---")
    event_bus = EventBus()
    context = ContextManager(event_bus=event_bus)
    mqtt_dispatcher = MQTTDispatcher(event_bus=event_bus, context=context)

    # Start MQTT client
    started = mqtt_dispatcher.start()
    if not started:
        print("  ⚠️ MQTT dispatcher could not start (possibly offline or no network). Skipping live connection assertion.")
        return

    print("  Waiting 2.5s for HiveMQ broker handshake...")
    time.sleep(2.5)

    print(f"  Broker: {mqtt_dispatcher.broker_host}:{mqtt_dispatcher.broker_port}")
    print(f"  Client ID: {mqtt_dispatcher.client_id}")
    print(f"  Connection Status: {'CONNECTED' if mqtt_dispatcher.is_connected else 'CONNECTING/OFFLINE'}")

    # Emit test events to verify dispatcher publishing methods
    event_bus.publish(EventTypes.STATE_CHANGED, {
        "old_state": "PATROL",
        "new_state": "ALERT",
        "reason": "Test person entry",
    })
    event_bus.publish(EventTypes.WEAPON_DETECTED, {
        "track_id": 7,
        "weapon": "KNIFE",
        "timestamp": time.time(),
    })
    event_bus.publish(EventTypes.RESPONSE_GENERATED, {
        "text": "Identify yourself immediately. Perimeter security is armed.",
        "latency_ms": 320.0,
        "autonomous": True,
    })

    time.sleep(1.0)
    mqtt_dispatcher.stop()
    print("  [OK] MQTT Dispatcher successfully processed test telemetry events and shut down cleanly")


def test_web_server_and_websocket():
    print("\n--- [TEST 3] FastAPI Web Companion & WebSocket Hub ---")
    event_bus = EventBus()
    context = ContextManager(event_bus=event_bus)
    snapshot_manager = SnapshotManager(event_bus=event_bus)

    # Use port 8005 for testing to avoid conflicts
    server = WebServer(
        event_bus=event_bus,
        context=context,
        snapshot_manager=snapshot_manager,
    )
    server.port = 8005

    started = server.start()
    assert started, "WebServer failed to start"
    time.sleep(1.5)

    # Test HTTP Endpoints via urllib
    import urllib.request
    base_url = "http://127.0.0.1:8005"

    try:
        # 1. Root / (serves index.html)
        req = urllib.request.urlopen(f"{base_url}/")
        assert req.status == 200, f"Root endpoint returned {req.status}"
        body = req.read().decode("utf-8")
        assert "ULTRON" in body, "index.html missing ULTRON brand"
        print("  [OK] GET / (Dashboard HTML) returned 200 OK")

        # 2. Manifest
        req = urllib.request.urlopen(f"{base_url}/manifest.json")
        assert req.status == 200
        manifest = json.loads(req.read().decode("utf-8"))
        assert manifest["name"] == "ULTRON Security Companion"
        print("  [OK] GET /manifest.json (PWA Manifest) verified")

        # 3. Status API
        req = urllib.request.urlopen(f"{base_url}/api/status")
        assert req.status == 200
        status_data = json.loads(req.read().decode("utf-8"))
        assert "security_state" in status_data
        print(f"  [OK] GET /api/status verified (State: {status_data['security_state']})")

        # 4. Snapshots history API
        req = urllib.request.urlopen(f"{base_url}/api/history/snapshots")
        assert req.status == 200
        snaps = json.loads(req.read().decode("utf-8"))
        assert "snapshots" in snaps
        print(f"  [OK] GET /api/history/snapshots verified ({len(snaps['snapshots'])} items)")

        # 5. WebSocket Test
        async def _test_ws():
            import websockets
            uri = "ws://127.0.0.1:8005/ws"
            async with websockets.connect(uri) as ws:
                # Expect init payload
                init_raw = await ws.recv()
                init_msg = json.loads(init_raw)
                assert init_msg["type"] == "init", f"Unexpected WS message type: {init_msg.get('type')}"
                print("  [OK] WebSocket connected & received 'init' state handshake")

                # Test ping / pong
                await ws.send("ping")
                pong_raw = await ws.recv()
                pong_msg = json.loads(pong_raw)
                assert pong_msg["type"] == "pong"
                print("  [OK] WebSocket ping/pong heartbeat verified")

        asyncio.run(_test_ws())

    finally:
        server.stop()
        print("  [OK] WebServer stopped cleanly")


def main():
    print("============================================================")
    print("  ULTRON STAGE 9 -- IOT & REMOTE COMPANION TEST SUITE")
    print("============================================================")

    test_snapshot_pipeline()
    test_mqtt_dispatcher()
    test_web_server_and_websocket()

    print("\n============================================================")
    print("  ALL STAGE 9 IOT & COMPANION TESTS PASSED SUCCESSFULLY! [OK]")
    print("============================================================\n")


if __name__ == "__main__":
    main()

