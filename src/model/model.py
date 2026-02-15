import joblib
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

import os
from huggingface_hub import hf_hub_download
import joblib

# On récupère le token hf
token = os.environ.get("HF_TOKEN")

def load_model():
    try:
        # On pointe vers le bon dépôt de modèle 
        model_path = hf_hub_download(
            repo_id="PCelia/credit-scoring-model", 
            filename="model.joblib",
            token=token
        )
        print("Modèle chargé avec succès depuis le Hub !")
        return joblib.load(model_path)
    except Exception as e:
        print(f"Échec HF Hub: {e}")
   

    # Local
    try:
        import mlflow.sklearn
        # On définit le chemin de DB mlflow relative à ce fichier
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "..", "..", "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
        model_uri = "models:/CreditScoring_LightGBM/Production"
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Échec chargement MLflow: {e}")

    raise FileNotFoundError("Impossible de charger le modèle (ni HF Hub, ni MLflow)")

