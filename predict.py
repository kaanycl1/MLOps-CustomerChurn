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
        log_file = os.path.join(os.getcwd(), "artifacts", "inference_logs.csv")
        log_dir = os.path.dirname(log_file)
        
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        X_logged = X.copy()
        X_logged["prediction_proba"] = proba
        X_logged["timestamp"] = pd.Timestamp.now()
        
        try:
            file_exists = os.path.exists(log_file)
            if not file_exists:
                X_logged.to_csv(log_file, index=False)
                print(f"Created inference log file: {log_file}")
            else:
                X_logged.to_csv(log_file, mode='a', header=False, index=False)
                print(f"Appended to inference log file: {log_file} ({len(X_logged)} rows)")
        except PermissionError as e:
            print(f"Permission error writing to {log_file}: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Log directory exists: {os.path.exists(log_dir)}, writable: {os.access(log_dir, os.W_OK)}")
        except Exception as e:
            print(f"Warning: Failed to log inference data to {log_file}: {e}")
            import traceback
            traceback.print_exc()
