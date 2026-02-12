import time
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


def test_response_time_under_threshold():
    client = get_client()

    start = time.time()
    response = client.post("/predict_by_id", json={"sk_id_curr": 100002})
    duration = time.time() - start

    assert response.status_code == 200
    assert duration < 1  # seuil large pour CI