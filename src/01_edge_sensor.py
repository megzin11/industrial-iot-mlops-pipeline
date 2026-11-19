import json
import os
import time
import uuid
import numpy as np
import paho.mqtt.client as mqtt

from src.detector import sample_telemetry_frame

MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "industrial/extruder-x1/telemetry")
MACHINE_ID = os.getenv("MACHINE_ID", "EXTRUDER-X1")
INTERVAL_SEC = float(os.getenv("PUBLISH_INTERVAL_SEC", "1.0"))


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[+] Edge Sensor connected to {MQTT_BROKER}:{MQTT_PORT} -> topic '{MQTT_TOPIC}'")
    else:
        print(f"[-] Connection failed with reason_code={reason_code}")


def main() -> None:
    client_id = f"edge-sensor-{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    client.on_connect = on_connect

    print(f"[*] Starting Industrial Edge Sensor Simulator (client_id={client_id})...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    rng = np.random.default_rng()
    try:
        while True:
            payload, regime = sample_telemetry_frame(rng, machine_id=MACHINE_ID, fault_prob=0.12)
            encoded = json.dumps(payload)
            client.publish(MQTT_TOPIC, encoded, qos=0)

            tag = "[NORMAL]" if regime == "normal" else f"[FAULT:{regime.upper()}]"
            print(f"{tag:<28} -> {encoded}")
            time.sleep(INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n[*] Shutting down Edge Sensor...")
    finally:
        client.loop_stop()
        client.disconnect()
        print("[+] Disconnected cleanly.")


if __name__ == "__main__":
    main()