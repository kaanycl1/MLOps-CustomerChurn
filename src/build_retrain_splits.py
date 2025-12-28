# src/build_retrain_splits.py
import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_TRAIN = "data/raw/customer_churn_dataset-training-master.csv"
FEEDBACK_PATH = "artifacts/feedback_store.csv"
OUT_DIR = "data/retrain_processed"

TARGET = "Churn"
ID_COL = "CustomerID"

RANDOM_STATE = 42


def main(test_size: float, val_size: float, random_state: int) -> None:
    if not os.path.exists(RAW_TRAIN):
        raise FileNotFoundError(f"Missing raw train file: {RAW_TRAIN}")

    base = pd.read_csv(RAW_TRAIN)

    if TARGET not in base.columns:
        raise ValueError(f"Base training CSV must include target column '{TARGET}'")

    # --- Base schema (THIS is the key fix) ---
    # We only allow columns that exist in the original base training dataset.
    base_cols = base.columns.tolist()

    # feedback is optional
    if os.path.exists(FEEDBACK_PATH):
        feedback = pd.read_csv(FEEDBACK_PATH)

        if TARGET not in feedback.columns:
            raise ValueError(f"Feedback store must include target column '{TARGET}'")

        # Keep only columns that match the base schema (drops feedback_ingested_at and any other junk)
        common_cols = [c for c in base_cols if c in feedback.columns]
        feedback = feedback[common_cols].copy()

        df = pd.concat([base[base_cols], feedback], ignore_index=True)

        # de-dup by ID if present
        if ID_COL in df.columns:
            df = df.drop_duplicates(subset=[ID_COL], keep="last")
    else:
        df = base[base_cols].copy()

    # --- Clean/validate target ---
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    before = len(df)
    df = df.dropna(subset=[TARGET])
    after = len(df)
    if after < before:
        print(f"[build_retrain_splits] Dropped {before-after} rows with invalid '{TARGET}'")

    df[TARGET] = df[TARGET].astype(int)

    # Split: test first, then val from remaining (val_size is fraction of full)
    strat = df[TARGET] if df[TARGET].nunique() > 1 else None

    trainval, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )

    strat_tv = trainval[TARGET] if trainval[TARGET].nunique() > 1 else None
    val_ratio_of_trainval = val_size / (1.0 - test_size)

    train, val = train_test_split(
        trainval,
        test_size=val_ratio_of_trainval,
        random_state=random_state,
        stratify=strat_tv
    )

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    stats = {
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "pos_rate_total": float(df[TARGET].mean()),
        "pos_rate_train": float(train[TARGET].mean()),
        "pos_rate_val": float(val[TARGET].mean()),
        "pos_rate_test": float(test[TARGET].mean()),
        "has_feedback": bool(os.path.exists(FEEDBACK_PATH)),
        "random_state": int(random_state),
        "test_size_full": float(test_size),
        "val_size_full": float(val_size),
        "schema_cols": base_cols,
    }
    pd.Series(stats).to_json(os.path.join(OUT_DIR, "retrain_stats.json"), indent=2)

    print(f"[build_retrain_splits] Wrote splits to {OUT_DIR} (total={len(df)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_size", type=float, default=0.20)
    ap.add_argument("--random_state", type=int, default=RANDOM_STATE)
    args = ap.parse_args()
    main(test_size=args.test_size, val_size=args.val_size, random_state=args.random_state)
