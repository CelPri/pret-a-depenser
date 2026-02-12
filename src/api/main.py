from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import os
import time
import json
from datetime import datetime
from huggingface_hub import hf_hub_download
from src.model.model import load_model

from fastapi.responses import RedirectResponse



class ClientID(BaseModel):
    sk_id_curr: int


from pathlib import Path
import os

def get_features_by_id(sk_id_curr: int) -> pd.DataFrame:
    # CAS TEST : features injectées par pytest
    if hasattr(app.state, "features") and app.state.features is not None:
        df = app.state.features

    else:
        # CAS LOCAL
        local_path = Path(__file__).resolve().parents[2] / "Data" / "features_clients.csv"
        if local_path.exists():
            df = pd.read_csv(local_path)

        else:
            # CAS HF : repo MODELE
            path = hf_hub_download(
                repo_id="PCelia/credit-scoring-model",
                filename="features_clients.csv",
                token=os.environ.get("HF_TOKEN")
            )
            df = pd.read_csv(path)

    row = df[df["SK_ID_CURR"] == sk_id_curr]
    if row.empty:
        raise KeyError("Client not found")

    return row.drop(columns=["SK_ID_CURR"])

def load_features() -> pd.DataFrame:
    # CAS LOCAL
    local_path = Path(__file__).resolve().parents[2] / "Data" / "features_clients.csv"
    if local_path.exists():
        return pd.read_csv(local_path)

    # CAS HF
    path = hf_hub_download(
        repo_id="PCelia/credit-scoring-model",
        filename="features_clients.csv",
        token=os.environ.get("HF_TOKEN")
    )
    return pd.read_csv(path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()

    # chargement unique des features
    app.state.features = load_features()

    yield


app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict_by_id")
def predict_by_id(payload: ClientID):
    start_total = time.perf_counter()

    try:
        start_features = time.perf_counter()
        X = get_features_by_id(payload.sk_id_curr)
        features_time = time.perf_counter() - start_features
    except KeyError:
        raise HTTPException(status_code=404, detail="Client not found")

    start_infer = time.perf_counter()
    score = float(app.state.model.predict_proba(X)[:, 1][0])
    infer_time = time.perf_counter() - start_infer

    total_time = time.perf_counter() - start_total

    print(
        f"features={features_time:.3f}s | "
        f"infer={infer_time:.3f}s | "
        f"total={total_time:.3f}s"
    )
    log_entry = {
    "timestamp": datetime.utcnow().isoformat(),
    "endpoint": "/predict_by_id",
    "sk_id_curr": payload.sk_id_curr,
    "score": score,
    "features_time": features_time,
    "inference_time": infer_time,
    "total_time": total_time
    }

    with open("api_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n") 
    
    return {
        "score": score,
        "features_time": features_time,
        "inference_time": infer_time,
        "total_time": total_time
    }

