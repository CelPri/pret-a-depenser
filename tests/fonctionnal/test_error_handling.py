from fastapi.testclient import TestClient
from src.api.main import app
import pandas as pd
import numpy as np


class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.4, 0.6]])


def get_client():
    app.state.model = DummyModel()
    app.state.features = pd.DataFrame({
        "SK_ID_CURR": [100002],
    })
    return TestClient(app)


def test_missing_field_returns_422():
    client = get_client()
    response = client.post("/predict_by_id", json={})
    assert response.status_code == 422


def test_wrong_type_returns_422():
    client = get_client()
    response = client.post("/predict_by_id", json={"sk_id_curr": "abc"})
    assert response.status_code == 422


def test_unknown_client_returns_404():
    client = get_client()
    response = client.post("/predict_by_id", json={"sk_id_curr": 999999})
    assert response.status_code == 404