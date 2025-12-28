from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import os
import json
import numpy as np
import pandas as pd

from config import TARGET, ID_COL, DROP_FEATURES

def split_xy(df):
    """Separate features and target"""
    y = df[TARGET].values
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    if DROP_FEATURES:
        X = X.drop(columns=[c for c in DROP_FEATURES if c in X.columns])
    return X, y

def evaluate(y_true, proba, threshold=0.5):
    """Standard binary classification metrics"""
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pred_pos_rate": pred.mean(),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist()
    }
