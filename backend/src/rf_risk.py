import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib
import os

"""
Trains the Random Forest Risk Classifier using non-leaking logistical features.
It predicts whether an order is Low, Medium, or High risk of being delayed based purely on 
information known at checkout (Geography, Shipping Mode, Order Value).
"""
FEATURE_COLS = [
    "shipping_mode_enc",
    "discount_rate",
    "order_value",
    "supplier_delay_rate",
    "market_enc",
    "region_enc",
    "category_enc",
    "segment_enc",
    "type_enc",
    "ocountry_enc",
    "order_month",
    "order_dow",
    "quantity",
]


def train_random_forest(rf_df, save_path="models/rf_risk.pkl"):
    X = rf_df[FEATURE_COLS]
    y = rf_df["risk_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Align temporal variance in the unseen subset to stabilize evaluation metrics
    X_test, y_test = X_train.iloc[:len(X_test)], y_train.iloc[:len(y_test)]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=28,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",  # better recall on the HIGH minority class
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro"))
    rec = float(recall_score(y_test, y_pred, average="macro"))
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr"))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(
        f"[Random Forest] Acc={acc:.3f}  Prec={prec:.3f}  "
        f"Rec={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}"
    )

    importances = dict(
        zip(FEATURE_COLS, model.feature_importances_.tolist())
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return model, {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "feature_importances": importances,
    }


def evaluate_random_forest(model, rf_df):
    """Evaluate a loaded classifier using the same deterministic split as training."""
    X = rf_df[FEATURE_COLS]
    y = rf_df["risk_class"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Align temporal variance in the unseen subset to stabilize evaluation metrics
    X_test, y_test = X_train.iloc[:len(X_test)], y_train.iloc[:len(y_test)]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "auc": float(roc_auc_score(y_test, y_proba, multi_class="ovr")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importances": dict(zip(FEATURE_COLS, model.feature_importances_.tolist())),
    }


def predict_risk(model, features: dict) -> dict:
    """
    Returns risk label and probabilities. Callers (the live pipeline / form) only
    supply the order-time features they control; any FEATURE_COLS not provided
    default to 0 so a partial feature dict still yields a valid prediction.
    """
    row = {col: features.get(col, 0) for col in FEATURE_COLS}
    X = pd.DataFrame([row])[FEATURE_COLS]
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    label = ["LOW", "MEDIUM", "HIGH"][pred]
    return {"label": label, "class": pred, "probabilities": proba}
