import joblib
from pathlib import Path

def load_model():
    # HF Space
    hf_path = Path("model.joblib")
    if hf_path.exists():
        return joblib.load(hf_path)

    # Local
    local_path = Path(__file__).resolve().parents[2] / "app" / "model.joblib"
    if local_path.exists():
        return joblib.load(local_path)

    raise FileNotFoundError("model.joblib not found")




# import mlflow
# import mlflow.sklearn
# import os


# current_dir = os.path.dirname(os.path.abspath(__file__))

# db_path = os.path.join(current_dir, "..", "..", "mlflow.db")

# mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# def load_model():
#     model_uri = "models:/CreditScoring_LightGBM/Production"
#     return mlflow.sklearn.load_model(model_uri)
