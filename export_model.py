import mlflow
import mlflow.sklearn
import joblib
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    "models:/CreditScoring_LightGBM/Production"
)

os.makedirs("app", exist_ok=True)
joblib.dump(model, "app/model.joblib")

print("OK")
