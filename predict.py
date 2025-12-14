# predict.py
import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/catboost_model.cbm")
DEFAULT_META_PATH  = os.getenv("META_PATH",  "artifacts/model_meta.joblib")

class ChurnPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, meta_path=DEFAULT_META_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta not found: {meta_path}")

        self.model_path = model_path
        self.meta_path = meta_path

        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

        meta = joblib.load(meta_path)
        self.cat_cols = meta.get("cat_cols", [])
        self.drop_features = meta.get("drop_features", [])
        self.target = meta.get("target", "Churn")

    def _normalize_input(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # drop target if someone accidentally sends it
        if self.target in X.columns:
            X = X.drop(columns=[self.target], errors="ignore")

        # drop columns we decided to drop in training
        if self.drop_features:
            X = X.drop(columns=[c for c in self.drop_features if c in X.columns], errors="ignore")

        # CatBoost likes consistent types for categoricals
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("string")

        return X

    def predict_proba(self, X: pd.DataFrame):
        Xn = self._normalize_input(X)
        proba = self.model.predict_proba(Xn)[:, 1]
        return proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5):
        proba = self.predict_proba(X)
        pred = (proba >= threshold).astype(int)
        return pred, proba
