import json
import numpy as np
import random
import paho.mqtt.client as mqtt
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.ensemble import IsolationForest

# --- CONFIGURATION ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "bepress/factory/machine_01/telemetry"
DB_PATH = "sqlite:///warehouse.db"

# --- DATABASE WAREHOUSE SETUP (SQLAlchemy ORM) ---
Base = declarative_base()

class Telemetry(Base):
    __tablename__ = 'telemetry_stream'
    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String)
    timestamp = Column(String)
    roller_temp_c = Column(Float)
    vibration_hz = Column(Float)
    is_anomaly = Column(Boolean)

# Initialize Database
engine = create_engine(DB_PATH, echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db_session = Session()

# --- MLOps: AI MODEL INITIALIZATION ---
print("[*] Training Isolation Forest model on baseline operational data...")
# We pre-train the model with 200 data points of "normal" BE PRESS factory behavior
baseline_data = np.array([[random.uniform(60.0, 75.0), random.uniform(10.0, 15.0)] for _ in range(200)])
ml_model = IsolationForest(contamination=0.05, random_state=42)
ml_model.fit(baseline_data)
print("[+] AI Model initialized and ready for live stream inference.")

# --- MQTT SUBSCRIBER ---
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[+] Connected to Broker. Listening to topic: {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"[-] Connection failed with code {reason_code}")

def on_message(client, userdata, msg):
    """Triggered every time a new JSON payload arrives from the Edge Sensor."""
    try:
        # 1. Parse the incoming JSON
        payload = json.loads(msg.payload.decode("utf-8"))
        temp = payload['roller_temp_c']
        vib = payload['vibration_hz']
        
        # 2. Run real-time AI Inference
        # IsolationForest predicts 1 for normal, -1 for anomaly
        prediction = ml_model.predict([[temp, vib]])[0] 
        ai_flagged_anomaly = bool(prediction == -1)
        
        # 3. Transform to ORM Object
        record = Telemetry(
            machine_id=payload['machine_id'],
            timestamp=payload['timestamp'],
            roller_temp_c=temp,
            vibration_hz=vib,
            is_anomaly=ai_flagged_anomaly
        )
        
        # 4. Load into Database
        db_session.add(record)
        db_session.commit()
        
        # 5. Terminal Feedback
        if ai_flagged_anomaly:
            print(f"🛑 [AI INTERCEPT] Anomaly saved to DB! Temp: {temp}C, Vib: {vib}Hz")
        else:
            print(f"[✓] Data warehoused -> {payload['machine_id']} | Normal")
            
    except Exception as e:
        print(f"[-] ETL Pipeline Error: {e}")
        db_session.rollback()

if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "BEPRESS_MLOps_Consumer")
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"[*] Connecting to {MQTT_BROKER}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down MLOps Consumer...")
        db_session.close()
        client.disconnect()