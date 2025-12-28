# src/collect_feedback.py
import argparse
import json
import os
from pathlib import Path

import pandas as pd


RAW_TEST = "data/raw/customer_churn_dataset-testing-master.csv"
STATE_PATH = "artifacts/retrain_state.json"
FEEDBACK_PATH = "artifacts/feedback_store.csv"


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "test_offset": 0,
        "total_feedback_rows": 0,
        "last_retrain_feedback_rows": 0,
        "model_version": 0,
    }


def save_state(path: str, state: dict) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def main(batch_size: int) -> None:
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    if not os.path.exists(RAW_TEST):
        raise FileNotFoundError(f"Missing raw test file: {RAW_TEST}")

    state = load_state(STATE_PATH)
    offset = int(state.get("test_offset", 0))

    df_test = pd.read_csv(RAW_TEST)
    n_total = len(df_test)

    if offset >= n_total:
        print(f"[collect_feedback] No new rows left. offset={offset}, total={n_total}")
        return

    end = min(offset + batch_size, n_total)
    batch = df_test.iloc[offset:end].copy()

    if "Churn" not in batch.columns:
        raise ValueError("Testing CSV must include 'Churn' for labeled feedback simulation.")

    batch["feedback_ingested_at"] = pd.Timestamp.now()

    # append to feedback store
    file_exists = os.path.exists(FEEDBACK_PATH)
    if not file_exists:
        batch.to_csv(FEEDBACK_PATH, index=False)
    else:
        batch.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)

    # update state
    state["test_offset"] = end
    # recompute total feedback rows from file (robust)
    state["total_feedback_rows"] = int(pd.read_csv(FEEDBACK_PATH).shape[0])
    save_state(STATE_PATH, state)

    print(
        f"[collect_feedback] Appended rows {offset}:{end} (n={len(batch)}) to {FEEDBACK_PATH}. "
        f"New offset={end}/{n_total}, total_feedback_rows={state['total_feedback_rows']}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()
    main(batch_size=args.batch_size)
