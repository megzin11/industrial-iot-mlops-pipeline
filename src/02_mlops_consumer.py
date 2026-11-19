import json
import os
import uuid
import paho.mqtt.client as mqtt

from src.detector import ExplainableAnomalyDetector
from src.models import DEFAULT_DB_URL, TelemetryRecord, create_session_factory

MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "industrial/extruder-x1/telemetry")


def main() -> None:
    _, SessionLocal = create_session_factory(DEFAULT_DB_URL)
    detector = ExplainableAnomalyDetector.load_or_train()
    print(f"[+] Loaded model '{detector.model_version}' | Warehouse: {DEFAULT_DB_URL}")

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print(f"[+] Subscribed to '{MQTT_TOPIC}' on {MQTT_BROKER}:{MQTT_PORT}")
            client.subscribe(MQTT_TOPIC)
        else:
            print(f"[-] Connection failed with reason_code={reason_code}")

    def on_message(client, userdata, msg):
        session = SessionLocal()
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            temp = float(payload["roller_temp_c"])
            vib = float(payload["vibration_hz"])
            current = float(payload["motor_current_a"])

            res = detector.score_reading(temp, vib, current)
            record = TelemetryRecord(
                machine_id=str(payload["machine_id"]),
                timestamp=str(payload["timestamp"]),
                roller_temp_c=temp,
                vibration_hz=vib,
                motor_current_a=current,
                anomaly_score=res.anomaly_score,
                is_anomaly=res.is_anomaly,
                primary_driver=res.primary_driver,
                inference_ms=res.inference_ms,
                model_version=res.model_version,
            )
            session.add(record)
            session.commit()

            if res.is_anomaly:
              print(
                  f"[ALERT] {record.machine_id} | driver={res.primary_driver:<18} | "
                  f"d_M={res.anomaly_score:>6.2f} | {res.inference_ms:.3f} ms"
              )
            else:
              print(
                  f"[OK]    {record.machine_id} | T={temp:.1f}C V={vib:.1f}Hz I={current:.1f}A | "
                  f"d_M={res.anomaly_score:>5.2f} | {res.inference_ms:.3f} ms"
              )
        except Exception as exc:
            session.rollback()
            print(f"[-] Pipeline ingestion error: {exc}")
        finally:
            session.close()

    client_id = f"mlops-consumer-{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[*] Connecting MLOps Consumer (client_id={client_id})...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down MLOps Consumer...")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()