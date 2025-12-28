from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from predict import ChurnPredictor
from fastapi.middleware.cors import CORSMiddleware

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = ChurnPredictor()
    yield
    predictor = None

app = FastAPI(title="Customer Churn API", version="1.1 (Phase 4)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]
    threshold: Optional[float] = 0.5

# phase 4: açıklamaları içerecek şekilde güncellendi 
class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    explanations: List[Dict[str, float]] 

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must be a non-empty list")

    if predictor is None:
        raise HTTPException(status_code=500, detail="model yüklenemedi")

    try:
        X = pd.DataFrame(req.rows)
        
        # 1. tahminleri al
        preds, probas = predictor.predict(X, threshold=float(req.threshold))
        
        # 2. phase 4: shap açıklamalarını al 
        shap_df = predictor.explain(X)
        explanations = shap_df.to_dict(orient="records")
        
        # 3. phase 4: drift analizi için logla [cite: 58]
        predictor.log_inference_data(X, probas)
        
        return PredictResponse(
            predictions=[int(x) for x in preds],
            probabilities=[float(x) for x in probas],
            explanations=explanations
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))