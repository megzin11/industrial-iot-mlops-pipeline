import time
import json
import random
import paho.mqtt.client as mqtt

# --- CONFIGURATION ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "bepress/factory/machine_01/telemetry" 

def generate_telemetry():
    """Generates synthetic machine data with occasional severe anomalies."""
    is_failing = random.random() > 0.90 

    if not is_failing:
        temp = round(random.uniform(60.0, 75.0), 2)
        vib = round(random.uniform(10.0, 15.0), 2)
    else:
        temp = round(random.uniform(95.0, 110.0), 2)
        vib = round(random.uniform(35.0, 50.0), 2)

    payload = {
        "machine_id": "EXTRUDER-X1",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "roller_temp_c": temp,
        "vibration_hz": vib
    }
    return payload, is_failing

# --- FIX: Updated v2 connection signature ---
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("[+] Edge Sensor successfully connected to the MQTT Broker.")
    else:
        print(f"[-] Connection failed with code {reason_code}")

if __name__ == "__main__":
    print("[*] Initializing Industrial Edge Sensor Simulator...")
    
    # --- FIX: Explicitly declare the v2 API version ---
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "BEPRESS_Edge_Producer")
    client.on_connect = on_connect
    
    print(f"[*] Connecting to {MQTT_BROKER}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    try:
        while True:
            payload, is_anomaly = generate_telemetry()
            json_payload = json.dumps(payload)
            
            client.publish(MQTT_TOPIC, json_payload)
            
            if is_anomaly:
                print(f"⚠️ [ANOMALY GENERATED] -> {json_payload}")
            else:
                print(f"[+] Streaming normal telemetry -> {json_payload}")
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[*] Shutting down Edge Sensor...")
        client.loop_stop()
        client.disconnect()
        print("[+] Disconnected cleanly.")