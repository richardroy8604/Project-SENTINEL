"""
ULTRON IoT — MQTT Telemetry & Incident Dispatcher
=================================================
Connects ULTRON Edge to an MQTT IoT Broker (HiveMQ Cloud, Mosquitto, or Local),
publishing structured JSON telemetry, live chat turns, security alerts, and
forensic Base64 snapshot payloads.

MQTT Topic Architecture:
  - ultron/telemetry/state     -> Live security state, active visitors, dwell times (QoS 0)
  - ultron/telemetry/chat      -> Conversational exchange turns (Intruder vs ULTRON) (QoS 1)
  - ultron/telemetry/alerts    -> High-priority security anomaly notifications (QoS 1)
  - ultron/telemetry/snapshot  -> Forensic Base64 JPEG snapshot with incident metadata (QoS 1)
  - ultron/telemetry/heartbeat -> Edge node liveness & system health (every 5s) (QoS 0)
"""

import json
import threading
import time
from typing import Optional

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

import config
from core.event_bus import EventBus, EventTypes, Event
from core.context import ContextManager


class MQTTDispatcher:
    """
    Edge IoT gateway publisher bridging ULTRON events to an MQTT Broker.
    """

    def __init__(self, event_bus: EventBus, context: ContextManager):
        self.event_bus = event_bus
        self.context = context

        self.enabled = getattr(config, "MQTT_ENABLED", True)
        self.broker_host = getattr(config, "MQTT_BROKER_HOST", "broker.hivemq.com")
        self.broker_port = getattr(config, "MQTT_BROKER_PORT", 1883)
        self.client_id = getattr(config, "MQTT_CLIENT_ID", f"ultron_edge_{int(time.time())}")
        self.topic_prefix = getattr(config, "MQTT_TOPIC_PREFIX", "ultron").rstrip("/")
        self.keepalive = getattr(config, "MQTT_KEEPALIVE", 60)

        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._start_time = time.time()

        if self.enabled and PAHO_AVAILABLE:
            self._setup_client()
            self._subscribe_events()

    def _setup_client(self):
        """Initialize Paho MQTT client."""
        try:
            # Use CallbackAPIVersion.VERSION2 for paho-mqtt >= 2.0
            self._client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
                clean_session=True,
            )
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

            # Set Will (LWT) message in case of ungraceful disconnection
            will_payload = json.dumps({
                "status": "OFFLINE",
                "client_id": self.client_id,
                "timestamp": time.time(),
                "reason": "Unexpected edge power down or network loss",
            })
            self._client.will_set(
                f"{self.topic_prefix}/telemetry/heartbeat",
                will_payload,
                qos=1,
                retain=True,
            )

        except Exception as e:
            print(f"[ULTRON MQTT] Error setting up client: {e}")
            self._client = None

    def _subscribe_events(self):
        """Wire event bus subscriptions to MQTT publication."""
        self.event_bus.subscribe(EventTypes.STATE_CHANGED, self._on_state_changed)
        self.event_bus.subscribe(EventTypes.SPEECH_RECOGNIZED, self._on_speech_recognized)
        self.event_bus.subscribe(EventTypes.RESPONSE_GENERATED, self._on_response_generated)
        self.event_bus.subscribe(EventTypes.WEAPON_DETECTED, self._on_weapon_detected)
        self.event_bus.subscribe(EventTypes.CAMERA_OBSTRUCTED, self._on_camera_obstructed)
        self.event_bus.subscribe(EventTypes.LOITERING_DETECTED, self._on_loitering_detected)
        self.event_bus.subscribe(EventTypes.SNAPSHOT_CAPTURED, self._on_snapshot_captured)

    def start(self) -> bool:
        """Start the background network loop and telemetry heartbeat."""
        if not self.enabled:
            print("[ULTRON MQTT] MQTT dispatch disabled in config.")
            return False

        if not PAHO_AVAILABLE or not self._client:
            print("[ULTRON MQTT] Warning: paho-mqtt not available. IoT dispatch skipped.")
            return False

        try:
            print(f"[ULTRON MQTT] Connecting to broker {self.broker_host}:{self.broker_port}...")
            self._client.connect_async(self.broker_host, self.broker_port, self.keepalive)
            self._client.loop_start()

            self._running = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="UltronMQTTHeartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()
            return True

        except Exception as e:
            print(f"[ULTRON MQTT] Connection error: {e}")
            return False

    def stop(self):
        """Gracefully disconnect and terminate loop."""
        self._running = False
        if self._client:
            try:
                # Publish graceful offline status
                offline_msg = json.dumps({
                    "status": "SHUTDOWN",
                    "client_id": self.client_id,
                    "timestamp": time.time(),
                })
                self._client.publish(
                    f"{self.topic_prefix}/telemetry/heartbeat", offline_msg, qos=1, retain=True
                )
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
        print("[ULTRON MQTT] IoT MQTT dispatcher stopped.")

    # ── Callbacks ───────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self._connected = True
            print(f"[ULTRON MQTT] Connected to broker: {self.broker_host}:{self.broker_port} (Topic prefix: '{self.topic_prefix}/#')")
            # Publish initial online status
            online_payload = json.dumps({
                "status": "ONLINE",
                "client_id": self.client_id,
                "timestamp": time.time(),
                "node": "ULTRON-EDGE-01",
                "topics": [
                    f"{self.topic_prefix}/telemetry/state",
                    f"{self.topic_prefix}/telemetry/chat",
                    f"{self.topic_prefix}/telemetry/alerts",
                    f"{self.topic_prefix}/telemetry/snapshot",
                ],
            })
            self._publish("telemetry/heartbeat", online_payload, qos=1, retain=True)
        else:
            self._connected = False
            print(f"[ULTRON MQTT] Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        self._connected = False
        if rc != 0:
            print(f"[ULTRON MQTT] Disconnected from broker (rc={rc}). Auto-reconnecting in background...")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Publication Helper ──────────────────────────────────────────

    def _publish(self, subtopic: str, payload: str, qos: int = 0, retain: bool = False):
        if not self._client or not self._running:
            return
        topic = f"{self.topic_prefix}/{subtopic.lstrip('/')}"
        try:
            self._client.publish(topic, payload, qos=qos, retain=retain)
        except Exception as e:
            print(f"[ULTRON MQTT] Publish error on {topic}: {e}")

    # ── Event Bus Handlers -> MQTT Topics ───────────────────────────

    def _on_state_changed(self, event: Event):
        """Publish security state transitions to ultron/telemetry/state."""
        data = {
            "timestamp": event.timestamp,
            "security_state": event.data.get("new_state", "UNKNOWN"),
            "old_state": event.data.get("old_state", "UNKNOWN"),
            "reason": event.data.get("reason", ""),
            "person_count": self.context.person_count,
            "persons": [
                {
                    "track_id": p.track_id,
                    "dwell_s": round(p.dwell_time, 1),
                    "holding_phone": p.holding_phone,
                    "holding_weapon": p.holding_weapon,
                    "weapon_type": p.weapon_type,
                }
                for p in self.context.persons
            ],
        }
        self._publish("telemetry/state", json.dumps(data), qos=0)

    def _on_speech_recognized(self, event: Event):
        """Publish visitor speech to ultron/telemetry/chat."""
        data = {
            "timestamp": event.timestamp,
            "role": "visitor",
            "text": event.data.get("text", ""),
            "latency_ms": event.data.get("latency_ms", 0.0),
        }
        self._publish("telemetry/chat", json.dumps(data), qos=1)

    def _on_response_generated(self, event: Event):
        """Publish ULTRON's verbal replies to ultron/telemetry/chat."""
        data = {
            "timestamp": event.timestamp,
            "role": "ultron",
            "text": event.data.get("text", ""),
            "latency_ms": event.data.get("latency_ms", 0.0),
            "autonomous": event.data.get("autonomous", False),
        }
        self._publish("telemetry/chat", json.dumps(data), qos=1)

    def _on_weapon_detected(self, event: Event):
        """Publish critical armed threat notifications to ultron/telemetry/alerts."""
        data = {
            "timestamp": event.timestamp,
            "alert_type": "ARMED_THREAT",
            "severity": "CRITICAL",
            "track_id": event.data.get("track_id", -1),
            "weapon": event.data.get("weapon", "WEAPON"),
            "action_required": "IMMEDIATE EVACUATION & LAW ENFORCEMENT NOTIFICATION",
        }
        self._publish("telemetry/alerts", json.dumps(data), qos=1)

    def _on_camera_obstructed(self, event: Event):
        """Publish camera tampering alerts to ultron/telemetry/alerts."""
        data = {
            "timestamp": event.timestamp,
            "alert_type": "CAMERA_TAMPER",
            "severity": "CRITICAL",
            "tamper_type": event.data.get("type", "LENS_COVERED"),
        }
        self._publish("telemetry/alerts", json.dumps(data), qos=1)

    def _on_loitering_detected(self, event: Event):
        """Publish loitering anomalies to ultron/telemetry/alerts."""
        data = {
            "timestamp": event.timestamp,
            "alert_type": "LOITERING",
            "severity": "WARNING",
            "track_id": event.data.get("track_id", -1),
            "duration_s": event.data.get("duration", 90.0),
        }
        self._publish("telemetry/alerts", json.dumps(data), qos=1)

    def _on_snapshot_captured(self, event: Event):
        """Publish high-res Base64 forensic snapshots to ultron/telemetry/snapshot."""
        data = {
            "timestamp": event.data.get("timestamp", time.time()),
            "time_str": event.data.get("time_str", ""),
            "trigger": event.data.get("trigger", "INCIDENT"),
            "security_state": event.data.get("security_state", ""),
            "has_weapon": event.data.get("has_weapon", False),
            "filename": event.data.get("filename", ""),
            "image_base64": event.data.get("base64", ""),
        }
        self._publish("telemetry/snapshot", json.dumps(data), qos=1)

    # ── Heartbeat Telemetry Loop ────────────────────────────────────

    def _heartbeat_loop(self):
        """Periodic status update sent every 5 seconds."""
        while self._running:
            try:
                time.sleep(5.0)
                if not self._running or not self._connected:
                    continue

                heartbeat_data = {
                    "timestamp": time.time(),
                    "uptime_s": round(time.time() - self._start_time, 1),
                    "status": "HEALTHY",
                    "security_state": self.context._security_state,
                    "person_count": self.context.person_count,
                    "connected_clients": 1,
                }
                self._publish("telemetry/heartbeat", json.dumps(heartbeat_data), qos=0)

            except Exception as e:
                print(f"[ULTRON MQTT] Heartbeat error: {e}")
