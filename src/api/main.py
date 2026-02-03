from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path

from src.model.model import load_model


BASE_DIR = Path(__file__).resolve().parents[2]


class ClientID(BaseModel):
    sk_id_curr: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    app.state.model = load_model()

    # Load features
    app.state.features = pd.read_csv(
        BASE_DIR / "Data" / "features_clients.csv"
    )

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict_by_id")
def predict_by_id(payload: ClientID):
    df = app.state.features

    row = df[df["SK_ID_CURR"] == payload.sk_id_curr]

    if row.empty:
        raise HTTPException(status_code=404, detail="Client not found")

    # IMPORTANT : retirer l’ID (pas utilisé au training)
    row = row.drop(columns=["SK_ID_CURR"])

    try:
        score = float(app.state.model.predict_proba(row)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"score": score}
