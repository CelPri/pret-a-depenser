import joblib
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

import os
from huggingface_hub import hf_hub_download
import joblib

# On récupère le token que tu vas créer dans les réglages du Space
token = os.environ.get("HF_TOKEN")

def load_model():
    try:
        # On pointe vers le bon dépôt de modèle (pas le space)
        model_path = hf_hub_download(
            repo_id="PCelia/credit-scoring-model", 
            filename="model.joblib",
            token=token
        )
        print("Modèle chargé avec succès depuis le Hub !")
        return joblib.load(model_path)
    except Exception as e:
        print(f"Échec HF Hub: {e}")
   

    # --- STRATÉGIE 2 : MLflow (Local uniquement) ---
    # Si on est chez toi, on utilise MLflow
    try:
        import mlflow.sklearn
        # On définit le chemin de ta DB mlflow relative à ce fichier
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "..", "..", "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
        model_uri = "models:/CreditScoring_LightGBM/Production"
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Échec chargement MLflow: {e}")

    raise FileNotFoundError("Impossible de charger le modèle (ni HF Hub, ni MLflow)")

# import joblib
# from pathlib import Path

# def load_model():
#     # HF Space
#     hf_path = Path("model.joblib")
#     if hf_path.exists():
#         return joblib.load(hf_path)

#     # Local
#     local_path = Path(__file__).resolve().parents[2] / "app" / "model.joblib"
#     if local_path.exists():
#         return joblib.load(local_path)

#     raise FileNotFoundError("model.joblib not found")




# # import mlflow
# # import mlflow.sklearn
# # import os


# # current_dir = os.path.dirname(os.path.abspath(__file__))

# # db_path = os.path.join(current_dir, "..", "..", "mlflow.db")

# # mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# # def load_model():
# #     model_uri = "models:/CreditScoring_LightGBM/Production"
# #     return mlflow.sklearn.load_model(model_uri)
