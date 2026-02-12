import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager

from src.api.main import app


# ---- Neutralise le lifespan ----
@asynccontextmanager
async def empty_lifespan(app):
    yield

app.router.lifespan_context = empty_lifespan
# --------------------------------


class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def get_client():
    app.state.model = DummyModel()
    app.state.features = pd.DataFrame({
        "SK_ID_CURR": [100002],
        "feature_1": [0.5],
        "feature_2": [1.2],
    })
    return TestClient(app)


def test_predict_existing_client():
    client = get_client()
    response = client.post("/predict_by_id", json={"sk_id_curr": 100002})
    assert response.status_code == 200
    assert "score" in response.json()


def test_predict_unknown_client():
    client = get_client()
    response = client.post("/predict_by_id", json={"sk_id_curr": 999999})
    assert response.status_code == 404
