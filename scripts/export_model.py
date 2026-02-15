import os
import mlflow
import mlflow.sklearn
import joblib


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

MODEL_URI = "models:/CreditScoring_LightGBM/Production"
OUT_PATH = os.path.join(BASE_DIR, "app", "model.joblib")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

model = mlflow.sklearn.load_model(MODEL_URI)
joblib.dump(model, OUT_PATH)

print("Export OK ->", OUT_PATH)
