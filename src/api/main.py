from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import os

from huggingface_hub import hf_hub_download
from src.model.model import load_model

from fastapi.responses import RedirectResponse




class ClientID(BaseModel):
    sk_id_curr: int


from pathlib import Path

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



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict_by_id")
def predict_by_id(payload: ClientID):
    try:
        X = get_features_by_id(payload.sk_id_curr)
    except KeyError:
        raise HTTPException(status_code=404, detail="Client not found")

    score = float(app.state.model.predict_proba(X)[:, 1][0])
    return {"score": score}




# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from contextlib import asynccontextmanager
# import pandas as pd
# from pathlib import Path

# from src.model.model import load_model


# BASE_DIR = Path(__file__).resolve().parents[2]


# class ClientID(BaseModel):
#     sk_id_curr: int


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load model
#     app.state.model = load_model()

#     # Load features
#     app.state.features = pd.read_csv(
#         BASE_DIR / "Data" / "features_clients.csv"
#     )

#     yield


# app = FastAPI(lifespan=lifespan)


# @app.post("/predict_by_id")
# def predict_by_id(payload: ClientID):
#     df = app.state.features

#     row = df[df["SK_ID_CURR"] == payload.sk_id_curr]

#     if row.empty:
#         raise HTTPException(status_code=404, detail="Client not found")

#     # IMPORTANT : retirer l’ID (pas utilisé au training)
#     row = row.drop(columns=["SK_ID_CURR"])

#     try:
#         score = float(app.state.model.predict_proba(row)[:, 1][0])
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     return {"score": score}
