import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.catboost
import mlflow.sklearn
import mlflow.xgboost
import joblib

from config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    EXPERIMENT_NAME, OUT_DIR,
    TARGET, ID_COL, RANDOM_STATE,
    CAT_COLS, DROP_FEATURES
)
from utils import split_xy, evaluate

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    X_train, y_train = split_xy(df_train)
    X_val, y_val = split_xy(df_val)
    X_test, y_test = split_xy(df_test)

    cat_cols = [c for c in CAT_COLS if c in X_train.columns]

    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_encoded[cat_cols] = ordinal_encoder.fit_transform(X_train[cat_cols].astype(str))
    X_val_encoded[cat_cols] = ordinal_encoder.transform(X_val[cat_cols].astype(str))
    X_test_encoded[cat_cols] = ordinal_encoder.transform(X_test[cat_cols].astype(str))

    hyperparams = {
        "RandomForest": {"max_depth": [5, 10, 15, 20]},
        "CatBoost": {"learning_rate": [0.01, 0.05, 0.1]},
        "XGBoost": {"max_depth": [4, 6, 8, 10]}
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, param_grid in hyperparams.items():
        param_name = list(param_grid.keys())[0]
        param_values = param_grid[param_name]

        for param_value in param_values:
            run_name = f"{model_name}_{param_name}_{param_value}"
            
            if model_name == "CatBoost":
                model = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    iterations=2000,
                    learning_rate=param_value,
                    depth=6,
                    l2_leaf_reg=5,
                    random_seed=RANDOM_STATE,
                    early_stopping_rounds=100,
                    verbose=200
                )
            elif model_name == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=param_value,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            else:
                model = XGBClassifier(
                    n_estimators=2000,
                    learning_rate=0.05,
                    max_depth=param_value,
                    random_state=RANDOM_STATE,
                    eval_metric="auc",
                    early_stopping_rounds=100
                )

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "model": model_name,
                    param_name: param_value,
                    "drop_features": ",".join(DROP_FEATURES) if DROP_FEATURES else "NONE"
                })

                if model_name == "CatBoost":
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        cat_features=cat_cols,
                        use_best_model=True
                    )
                    test_proba = model.predict_proba(X_test)[:, 1]
                elif model_name == "XGBoost":
                    model.fit(
                        X_train_encoded, y_train,
                        eval_set=[(X_val_encoded, y_val)],
                        verbose=200
                    )
                    test_proba = model.predict_proba(X_test_encoded)[:, 1]
                else:
                    model.fit(X_train_encoded, y_train)
                    test_proba = model.predict_proba(X_test_encoded)[:, 1]

                metrics = evaluate(y_test, test_proba)

                print(f"\n{run_name} TEST METRICS @0.5")
                for k, v in metrics.items():
                    if k != "confusion_matrix":
                        print(f"{k}: {v}")
                print("Confusion matrix:")
                print(np.array(metrics["confusion_matrix"]))

                for k, v in metrics.items():
                    if k != "confusion_matrix":
                        mlflow.log_metric(k, v)

                report_path = os.path.join(OUT_DIR, f"{run_name.lower().replace('_', '-')}_test_report.json")
                with open(report_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                mlflow.log_artifact(report_path)

                meta_path = os.path.join(OUT_DIR, f"{run_name.lower().replace('_', '-')}_meta.joblib")
                meta = {
                    "cat_cols": cat_cols,
                    "drop_features": DROP_FEATURES,
                    "target": TARGET
                }
                if model_name != "CatBoost":
                    meta["ordinal_encoder"] = ordinal_encoder
                joblib.dump(meta, meta_path)
                mlflow.log_artifact(meta_path)

                if model_name == "CatBoost":
                    mlflow.catboost.log_model(model, artifact_path="model")
                elif model_name == "XGBoost":
                    mlflow.xgboost.log_model(model, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model")

    print("\nDONE. Using processed train/val/test splits + MLflow tracking.")

if __name__ == "__main__":
    main()