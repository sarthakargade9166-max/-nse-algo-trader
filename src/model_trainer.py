import numpy as np
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import streamlit as st
from src.feature_engineer import get_feature_columns

MODEL_PATH = "models/xgboost_nse_model.pkl"


def create_target_labels(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.005) -> pd.Series:
    """
    Creates the target label (what we want to predict).
    
    Logic:
        - Compute future return = (Close[t+horizon] - Close[t]) / Close[t]
        - If future_return >  threshold → BUY  (label = 1)
        - If future_return < -threshold → SELL (label = -1)
        - Otherwise                     → HOLD (label = 0)

    Parameters:
        df        (pd.DataFrame): DataFrame with 'Close' column
        horizon   (int):          How many days ahead to predict (1, 3, or 5)
        threshold (float):        Minimum % move to be classified as BUY/SELL (default 0.5%)

    Returns:
        pd.Series: Target labels (1, 0, or -1)
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1

    labels = pd.Series(0, index=df.index)               # Default: HOLD
    labels[future_return >  threshold] = 1               # BUY
    labels[future_return < -threshold] = -1              # SELL

    return labels


def train_xgboost_model(
    df: pd.DataFrame,
    n_estimators: int   = 200,
    max_depth: int      = 4,
    test_size: float    = 0.20,
    horizon: int        = 1
) -> tuple:
    """
    Full training pipeline:
        1. Create target labels
        2. Split data (time-series aware — NO SHUFFLE)
        3. Train XGBoost
        4. Evaluate on test set
        5. Save model to disk

    CRITICAL — Why no shuffle?
        Time-series data must NEVER be shuffled before splitting.
        If we train on future data and test on past data, we get
        "data leakage" — artificially inflated accuracy that won't
        hold in real trading.

    Parameters:
        df           (pd.DataFrame): Feature + regime engineered DataFrame
        n_estimators (int):          Number of XGBoost trees
        max_depth    (int):          Maximum depth of each tree
        test_size    (float):        Fraction of data to hold out for testing
        horizon      (int):          Prediction horizon in days

    Returns:
        (model, metrics_dict, X_test, y_test, y_pred)
    """
    # ─── Step 1: Create Target Labels ────────────────────────────────────
    labels = create_target_labels(df, horizon=horizon)

    # Align features and labels (remove last `horizon` rows — no future data)
    feature_cols = get_feature_columns() + ["Regime_Num", "ADX", "Regime_Confidence"]
    available_features = [c for c in feature_cols if c in df.columns]

    X = df[available_features].iloc[:-horizon]
    y = labels.iloc[:-horizon]

    # Remap labels: -1 → 0, 0 → 1, 1 → 2 (XGBoost needs 0-based integers)
    label_map    = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}
    y_encoded    = y.map(label_map)

    # step 2: Time-Series Train/Test Split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_encoded.iloc[:split_idx], y_encoded.iloc[split_idx:]

    # ─── Step 3: Train XGBoost
    model = XGBClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        learning_rate     = 0.05,         
        subsample         = 0.8,          
        colsample_bytree  = 0.8,          
        use_label_encoder = False,
        eval_metric       = "mlogloss",   
        random_state      = 42,
        n_jobs            = -1           
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Step 4: Evaluate 
    y_pred_encoded = model.predict(X_test)
    y_pred_proba   = model.predict_proba(X_test)
    y_pred = np.array([inv_label_map[p] for p in y_pred_encoded])
    y_test_orig = np.array([inv_label_map[p] for p in y_test])

    metrics = {
        "accuracy":  accuracy_score(y_test_orig,  y_pred),
        "precision": precision_score(y_test_orig, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test_orig,    y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test_orig,        y_pred, average="weighted", zero_division=0),
        "roc_auc":   roc_auc_score(y_test,        y_pred_proba, multi_class="ovr", average="weighted"),
        "report":    classification_report(y_test_orig, y_pred, target_names=["SELL", "HOLD", "BUY"], zero_division=0)
    }

    #  Step 5: Save Model 
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": available_features, "label_map": inv_label_map}, MODEL_PATH)

    return model, metrics, X_test, y_test_orig, y_pred


def load_model() -> dict | None:
    """Loads a previously saved model from disk."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict_signal(model, features: pd.DataFrame, feature_cols: list) -> int:
    """
    Generates a trading signal for a single row of features.

    Returns:
        int: 1 (BUY), 0 (HOLD), or -1 (SELL)
    """
    X = features[feature_cols].iloc[[-1]]  
    raw_pred = model.predict(X)[0]
    inv_map  = {0: -1, 1: 0, 2: 1}
    return inv_map.get(raw_pred, 0)
