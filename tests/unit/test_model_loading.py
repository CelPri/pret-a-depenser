import pytest
from unittest.mock import patch, MagicMock

from src.model.model import load_model


# HF OK
@patch("src.model.model.joblib.load")
@patch("src.model.model.hf_hub_download")
def test_load_model_from_hf(mock_hf, mock_joblib):
    mock_hf.return_value = "fake_path.joblib"
    mock_joblib.return_value = "MODEL"

    model = load_model()

    assert model == "MODEL"


# HF échoue → MLflow OK
@patch("src.model.model.hf_hub_download", side_effect=Exception("HF fail"))
@patch("mlflow.sklearn.load_model")
def test_load_model_fallback_mlflow(mock_mlflow, mock_hf):
    mock_mlflow.return_value = "MLFLOW_MODEL"

    model = load_model()

    assert model == "MLFLOW_MODEL"


# Tout échoue → FileNotFoundError
@patch("src.model.model.hf_hub_download", side_effect=Exception("HF fail"))
@patch("mlflow.sklearn.load_model", side_effect=Exception("MLflow fail"))
def test_load_model_raises_error(mock_mlflow, mock_hf):
    with pytest.raises(FileNotFoundError):
        load_model()