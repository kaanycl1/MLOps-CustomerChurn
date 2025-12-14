# api.py
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from predict import ChurnPredictor

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = ChurnPredictor()
    yield
    predictor = None

app = FastAPI(title="Customer Churn API", version="1.0", lifespan=lifespan)

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]
    threshold: Optional[float] = 0.5

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must be a non-empty list")

    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X = pd.DataFrame(req.rows)
        preds, probas = predictor.predict(X, threshold=float(req.threshold))
        return PredictResponse(
            predictions=[int(x) for x in preds],
            probabilities=[float(x) for x in probas]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
