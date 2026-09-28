"""
ULTRON Standalone IoT Subscriber
================================
A 100% decoupled, independent MQTT subscriber client.
Demonstrates to college evaluators that ULTRON operates under a true
Publish/Subscribe IoT architecture.

Can be run on this machine or ANY other computer anywhere in the world!
Requirements: pip install paho-mqtt
"""

import json
import ssl
import sys
import time
from pathlib import Path

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
except ImportError:
    print("Error: paho-mqtt is required. Run: pip install paho-mqtt")
    sys.exit(1)

# HiveMQ Cloud Private Cluster Configuration
BROKER_HOST = "2fd0cc9fccec48b9befe7b3d4b75cc59.s1.eu.hivemq.cloud"
BROKER_PORT = 8883
USERNAME = "hivemq.webclient.1790576042940"
PASSWORD = "nMJJC!GVDRqO*cKXz%Rhcyiwz7c6FPCq"
TOPIC_SUBSCRIPTION = "ultron/#"

CLIENT_ID = f"ultron_remote_monitor_{int(time.time())}"


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0 or str(rc) == "Success":
        print("=" * 65)
        print("  CONNECTED TO HIVEMQ CLOUD SECURE CLUSTER (TLS PORT 8883)")
        print(f"  Subscribed to topic pattern: {TOPIC_SUBSCRIPTION}")
        print("  Awaiting live security telemetry from ULTRON Edge...")
        print("=" * 65)
        client.subscribe(TOPIC_SUBSCRIPTION, qos=1)
    else:
        print(f"[ERROR] Connection failed with code: {rc}")


def on_message(client, userdata, msg):
    topic = msg.topic
    now_str = time.strftime("%H:%M:%S")

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        payload = {"raw": msg.payload.decode("utf-8", errors="ignore")}

    if "alerts" in topic:
        sev = payload.get("severity", "CRITICAL")
        alert_type = payload.get("event", payload.get("alert_type", "ALERT"))
        desc = payload.get("description", payload.get("weapon", ""))
        print(f"\n🚨 [{now_str}] [CRITICAL ALERT] {sev}: {alert_type} -> {desc}")

    elif "snapshot" in topic:
        fname = payload.get("filename", "snapshot.jpg")
        trig = payload.get("trigger", "INCIDENT")
        b64_len = len(payload.get("base64", ""))
        threat = payload.get("threat_level", "NOMINAL")
        print(f"\n📸 [{now_str}] [FORENSIC SNAPSHOT] {trig} | Threat: {threat} | File: {fname} ({b64_len} chars Base64)")

    elif "chat" in topic:
        role = payload.get("role", "SYSTEM").upper()
        text = payload.get("text", "")
        latency = payload.get("latency_ms", 0.0)
        print(f"\n💬 [{now_str}] [COMMS] {role}: \"{text}\" ({latency:.0f}ms)")

    elif "state" in topic:
        st = payload.get("security_state", "PATROL")
        cnt = payload.get("person_count", 0)
        print(f"[{now_str}] [STATE] Security: {st} | Visitors Detected: {cnt}")

    elif "heartbeat" in topic:
        status = payload.get("status", "ONLINE")
        uptime = payload.get("uptime_s", 0)
        print(f"[{now_str}] [HEARTBEAT] Edge Node Status: {status} (Uptime: {int(uptime)}s)")

    else:
        print(f"[{now_str}] [{topic}]: {json.dumps(payload, indent=2)}")


def main():
    print("=" * 65)
    print("  ULTRON STANDALONE IOT MONITORING CONSOLE")
    print(f"  Broker: {BROKER_HOST}:{BROKER_PORT}")
    print("=" * 65)
    print("Connecting over TLS...")

    client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION2,
        client_id=CLIENT_ID,
        clean_session=True,
    )
    client.tls_set()
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nDisconnecting from broker...")
        client.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
