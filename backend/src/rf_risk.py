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

FEATURE_COLS = [
    "shipping_mode_enc",
    "actual_days",
    "scheduled_days",
    "discount_rate",
    "order_value",
    "supplier_delay_rate",
    "days_buffer",
    "delay_gap",
]


def train_random_forest(rf_df, save_path="models/rf_risk.pkl"):
    X = rf_df[FEATURE_COLS]
    y = rf_df["risk_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
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
    """Returns risk label and probabilities."""
    X = pd.DataFrame([features])[FEATURE_COLS]
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    label = ["LOW", "MEDIUM", "HIGH"][pred]
    return {"label": label, "class": pred, "probabilities": proba}
