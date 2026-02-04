from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import os

from src.model.model import load_model


class ClientFeatures(BaseModel):
    features: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
def predict(payload: ClientFeatures):
    try:
        X = pd.DataFrame([payload.features])
        score = float(app.state.model.predict_proba(X)[:, 1][0])
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
