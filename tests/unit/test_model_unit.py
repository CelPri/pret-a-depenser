import numpy as np
import pandas as pd


class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


def test_predict_proba_output_valid():
    model = DummyModel()
    X = pd.DataFrame([[1, 2, 3]])

    proba = model.predict_proba(X)

    # shape correcte
    assert proba.shape == (1, 2)

    # valeurs entre 0 et 1
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)

    # pas de NaN
    assert not np.isnan(proba).any()