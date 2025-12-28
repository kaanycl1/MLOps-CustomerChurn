import os
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

# varsayılan dosya yolları artifacts klasörüne bakar 
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/catboost_model.cbm")
DEFAULT_META_PATH  = os.getenv("META_PATH",  "artifacts/model_meta.joblib")

class ChurnPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, meta_path=DEFAULT_META_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model bulunamadı: {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta verisi bulunamadı: {meta_path}")

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
        if self.target in X.columns:
            X = X.drop(columns=[self.target], errors="ignore")

        if self.drop_features:
            X = X.drop(columns=[c for c in self.drop_features if c in X.columns], errors="ignore")

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

    # phase 4: explainability (shap) 
    def explain(self, X: pd.DataFrame):
        Xn = self._normalize_input(X)
        pool = Pool(Xn, cat_features=self.cat_cols)
        # catboost içindeki shap değerlerini hesaplar 
        shap_values = self.model.get_feature_importance(pool, type='ShapValues')
        feature_names = Xn.columns.tolist()
        return pd.DataFrame(shap_values[:, :-1], columns=feature_names)

    # phase 4: monitoring log (evidently ai hazırlığı) [cite: 58]
    def log_inference_data(self, X: pd.DataFrame, proba: np.ndarray):
        log_file = "artifacts/inference_logs.csv"
        X_logged = X.copy()
        X_logged["prediction_proba"] = proba
        X_logged["timestamp"] = pd.Timestamp.now()
        
        if not os.path.exists(log_file):
            X_logged.to_csv(log_file, index=False)
        else:
            X_logged.to_csv(log_file, mode='a', header=False, index=False)