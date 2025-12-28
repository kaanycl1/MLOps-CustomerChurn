# src/retrain_gate.py
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

try:
    from config import TARGET, ID_COL, CAT_COLS, DROP_FEATURES, RANDOM_STATE
except Exception:
    TARGET = "Churn"
    ID_COL = "CustomerID"
    CAT_COLS = ["Gender", "Subscription Type", "Contract Length"]
    DROP_FEATURES = []
    RANDOM_STATE = 42

RETRAIN_TRAIN_CSV = "data/retrain_processed/train.csv"
RETRAIN_VAL_CSV   = "data/retrain_processed/val.csv"
RETRAIN_TEST_CSV  = "data/retrain_processed/test.csv"

STATE_PATH = "artifacts/retrain_state.json"
FEEDBACK_PATH = "artifacts/feedback_store.csv"

MODEL_LATEST = "artifacts/catboost_model.cbm"
META_LATEST  = "artifacts/model_meta.joblib"
RUN_REPORT   = "artifacts/retrain_run.json"


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"test_offset": 0, "total_feedback_rows": 0, "last_retrain_feedback_rows": 0, "model_version": 0}


def save_state(state: dict) -> None:
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def write_run_report(payload: dict) -> None:
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    with open(RUN_REPORT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def evaluate_basic(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (proba >= thr).astype(int)
    acc = float((y_pred == y_true).mean())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def count_feedback_rows() -> int:
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    return int(pd.read_csv(FEEDBACK_PATH).shape[0])


def main(retrain_threshold: int) -> None:
    state = load_state()

    total_feedback = count_feedback_rows()
    last_retrain_at = int(state.get("last_retrain_feedback_rows", 0))
    newly_arrived = total_feedback - last_retrain_at

    print(
        f"[retrain_gate] feedback_total={total_feedback}, last_retrain_at={last_retrain_at}, "
        f"new_since_last={newly_arrived}, threshold={retrain_threshold}"
    )

    # ALWAYS write a run report so DVC has an output even when we skip
    base_report = {
        "feedback_total": total_feedback,
        "last_retrain_at": last_retrain_at,
        "new_since_last": newly_arrived,
        "threshold": retrain_threshold,
        "action": "skip",
        "model_version_before": int(state.get("model_version", 0)),
    }

    if newly_arrived < retrain_threshold:
        print("[retrain_gate] Not enough new labeled data. Skipping retrain.")
        state["total_feedback_rows"] = total_feedback
        save_state(state)
        write_run_report(base_report)
        return

    # ensure splits exist
    for p in [RETRAIN_TRAIN_CSV, RETRAIN_VAL_CSV, RETRAIN_TEST_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}. Run build_retrain_splits first.")

    df_train = pd.read_csv(RETRAIN_TRAIN_CSV)
    df_val   = pd.read_csv(RETRAIN_VAL_CSV)
    df_test  = pd.read_csv(RETRAIN_TEST_CSV)

    def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if DROP_FEATURES:
            d = d.drop(columns=[c for c in DROP_FEATURES if c in d.columns], errors="ignore")
        return d

    df_train = drop_cols(df_train)
    df_val   = drop_cols(df_val)
    df_test  = drop_cols(df_test)

    # clean target
    for d in (df_train, df_val, df_test):
        d[TARGET] = pd.to_numeric(d[TARGET], errors="coerce")
        d.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
        d.dropna(subset=[TARGET], inplace=True)
        d[TARGET] = d[TARGET].astype(int)

    y_train = df_train[TARGET].values
    y_val   = df_val[TARGET].values
    y_test  = df_test[TARGET].values

    X_train = df_train.drop(columns=[TARGET], errors="ignore")
    X_val   = df_val.drop(columns=[TARGET], errors="ignore")
    X_test  = df_test.drop(columns=[TARGET], errors="ignore")

    cat_cols = [c for c in CAT_COLS if c in X_train.columns]
    for c in cat_cols:
        X_train[c] = X_train[c].astype("string")
        X_val[c]   = X_val[c].astype("string")
        X_test[c]  = X_test[c].astype("string")

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        random_seed=RANDOM_STATE,
        early_stopping_rounds=100,
        verbose=200,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_cols, use_best_model=True)

    test_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_basic(y_test, test_proba, thr=0.5)

    state["model_version"] = int(state.get("model_version", 0)) + 1
    version = state["model_version"]

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_LATEST)

    meta = {
        "cat_cols": cat_cols,
        "drop_features": DROP_FEATURES,
        "target": TARGET,
        "id_col": ID_COL,
        "trained_on_feedback_rows": total_feedback,
        "metrics_test": metrics,
        "model_version": version,
    }
    joblib.dump(meta, META_LATEST)

    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    model_v_path = f"artifacts/models/catboost_model_v{version:03d}.cbm"
    meta_v_path  = f"artifacts/models/model_meta_v{version:03d}.joblib"
    report_v_path = f"artifacts/models/retrain_report_v{version:03d}.json"

    model.save_model(model_v_path)
    joblib.dump(meta, meta_v_path)

    with open(report_v_path, "w", encoding="utf-8") as f:
        json.dump(
            {"version": version, "feedback_total": total_feedback, "new_since_last": newly_arrived, "metrics_test": metrics},
            f,
            indent=2,
        )

    state["total_feedback_rows"] = total_feedback
    state["last_retrain_feedback_rows"] = total_feedback
    save_state(state)

    # write run report for this execution
    write_run_report({
        **base_report,
        "action": "retrain",
        "model_version_after": version,
        "metrics_test": metrics,
        "model_latest": MODEL_LATEST,
        "model_versioned": model_v_path,
    })

    print(f"[retrain_gate] ✅ Retrained model v{version:03d}. Saved to {MODEL_LATEST} and {model_v_path}")
    print(f"[retrain_gate] Test metrics: {metrics}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain_threshold", type=int, default=500)
    args = ap.parse_args()
    main(retrain_threshold=args.retrain_threshold)
