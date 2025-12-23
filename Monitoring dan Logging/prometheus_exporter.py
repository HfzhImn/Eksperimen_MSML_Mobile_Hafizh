from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import requests
import random
import json

# Definisi Metrik
REQUEST_COUNT = Counter('request_count', 'Total Request Inference')
LATENCY = Summary('request_latency_seconds', 'Waktu Inference')
PREDICTION_GAUGE = Gauge('prediction_value', 'Prediksi Harga (0-3)')

def generate_full_dummy_data():
    # Generate data acak untuk 20 Fitur HP
    return [
        random.randint(500, 2000),      # battery_power
        random.randint(0, 1),           # blue (Bluetooth)
        random.uniform(0.5, 3.0),       # clock_speed
        random.randint(0, 1),           # dual_sim
        random.randint(0, 20),          # fc (Front Camera)
        random.randint(0, 1),           # four_g
        random.randint(2, 64),          # int_memory
        random.uniform(0.1, 1.0),       # m_dep (Depth)
        random.randint(80, 200),        # mobile_wt (Weight)
        random.randint(1, 8),           # n_cores
        random.randint(0, 20),          # pc (Primary Camera)
        random.randint(100, 1900),      # px_height
        random.randint(500, 1990),      # px_width
        random.randint(256, 4000),      # ram
        random.randint(5, 20),          # sc_h (Screen Height)
        random.randint(0, 18),          # sc_w (Screen Width)
        random.randint(2, 20),          # talk_time
        random.randint(0, 1),           # three_g
        random.randint(0, 1),           # touch_screen
        random.randint(0, 1)            # wifi
    ]

def monitor():
    endpoint = "http://127.0.0.1:5002/invocations"
    
    # Nama 20 Kolom (URUTAN WAJIB SAMA DENGAN DATASET)
    columns = [
        "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
        "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
        "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
        "touch_screen", "wifi"
    ]

    while True:
        data = generate_full_dummy_data()
        
        payload = {
            "dataframe_split": {
                "columns": columns,
                "data": [data]
            }
        }

        try:
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                pred = response.json()['predictions'][0]
                PREDICTION_GAUGE.set(pred)
                print(f"✅ Data Lengkap Terkirim! Prediksi Kelas: {pred}")
            else:
                print(f"⚠️ Error: {response.text}")
        except Exception as e:
            print(f"❌ Koneksi Error: {e}")

        time.sleep(2)

if __name__ == '__main__':
    print("🚀 Exporter Full-Feature Berjalan...")
    start_http_server(8000)
    monitor()