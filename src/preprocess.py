import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Churn"
ID_COL = "CustomerID"

CAT_COLS = ["Gender", "Subscription Type", "Contract Length"]
NUM_COLS = [
    "Age", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay",
    "Total Spend", "Last Interaction"
]

def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize churn labels to {0,1} and drop rows without valid target."""
    df = df.copy()
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataframe.")

    df[TARGET] = df[TARGET].replace({
        "Yes": 1, "No": 0,
        "yes": 1, "no": 0,
        True: 1, False: 0
    })
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])]
    return df

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic dtype cleanup for known categorical/numerical columns (if present)."""
    df = df.copy()
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def split_xy(df: pd.DataFrame, drop_features=None):
    """Separate features and target."""
    drop_features = drop_features or []
    y = df[TARGET].values
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    if drop_features:
        X = X.drop(columns=[c for c in drop_features if c in X.columns], errors="ignore")
    return X, y

def class_ratio(y: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    return {
        "n": int(len(y)),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
        "pos_rate": float((y == 1).mean()) if len(y) else 0.0
    }

def main():
    parser = argparse.ArgumentParser(description="Preprocess churn data and create train/val/test splits.")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV (raw).")
    parser.add_argument("--test_csv", required=True, help="Path to testing CSV (raw).")
    parser.add_argument("--out_dir", required=True, help="Output directory for processed splits.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.20, help="Test fraction of FULL dataset.")
    parser.add_argument("--val_size", type=float, default=0.20, help="Validation fraction of FULL dataset.")
    parser.add_argument("--drop_features", nargs="*", default=[], help="Optional feature columns to drop.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read and concatenate (as you requested)
    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Clean
    df = clean_target(df)      # ensures only labeled rows remain
    df = clean_features(df)

    # Split
    X, y = split_xy(df, drop_features=args.drop_features)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_tmp
    )

    # Save splits (keep target column)
    train_df = X_train.copy()
    train_df[TARGET] = y_train
    val_df = X_val.copy()
    val_df[TARGET] = y_val
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Write stats for CI sanity checks
    stats = {
        "inputs": {
            "train_csv": str(args.train_csv),
            "test_csv": str(args.test_csv),
            "rows_train_csv": int(len(df_train)),
            "rows_test_csv": int(len(df_test)),
            "rows_concat": int(len(df_train) + len(df_test)),
            "rows_after_label_clean": int(len(df)),
        },
        "splits": {
            "test_size_full": float(args.test_size),
            "val_size_full": float(args.val_size),
            "train_size_full": float(1 - args.test_size - args.val_size),
            "random_state": int(args.random_state),
        },
        "class_balance": {
            "train": class_ratio(y_train),
            "val": class_ratio(y_val),
            "test": class_ratio(y_test),
        },
        "outputs": {
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
        }
    }

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats["splits"], indent=2))
    print("Train:", stats["class_balance"]["train"])
    print("Val  :", stats["class_balance"]["val"])
    print("Test :", stats["class_balance"]["test"])

if __name__ == "__main__":
    main()
