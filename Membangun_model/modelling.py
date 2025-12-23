import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# 1. Load Dataset BERSIH 
# Mengambil data dari folder preprocessing
try:
    # Path naik satu level (..) lalu masuk ke preprocessing
    df = pd.read_csv('../preprocessing/mobile_price_clean.csv')
    print("✅ Dataset BERSIH (Clean) berhasil dimuat!")
except FileNotFoundError:
    print("❌ Error: File 'mobile_price_clean.csv' tidak ditemukan di folder preprocessing.")
    print("Harap jalankan Notebook preprocessing terlebih dahulu.")
    exit()

# 2. Feature Selection & Split Data
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Set Experiment Name
mlflow.set_experiment("Mobile_Price_Classification_Revisi")

# --- PERBAIKAN UTAMA: AUTOLOGGING (Sesuai Kriteria 2) ---
mlflow.sklearn.autolog()

print("Starting training with Autologging...")

with mlflow.start_run():
    # Define Hyperparameters
    n_estimators = 100
    max_depth = 10
    
    # Initialize Model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    # Train Model
    # Karena ada autolog(), MLflow otomatis mencatat:
    # - Parameter (n_estimators, max_depth)
    # - Metrics (accuracy, loss, dll)
    # - Model Artifact (file modelnya)
    model.fit(X_train, y_train)

    # Evaluasi Manual (Hanya untuk print di terminal, tidak perlu log ke mlflow lagi)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"✅ Model Training Selesai. Akurasi Test: {acc:.4f}")
    print("🚀 Semua metrics & model sudah tersimpan otomatis oleh Autolog.")