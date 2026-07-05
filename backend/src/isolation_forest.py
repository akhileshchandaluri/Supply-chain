import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

SUPPLIER_FEATURES = [
    "avg_delivery_time",
    "price_deviation",
    "fulfillment_rate",
    "complaint_freq",
]


def train_isolation_forest(supplier_df, save_path="models/isolation_forest.pkl"):
    X = supplier_df[SUPPLIER_FEATURES].fillna(0)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # expect ~5% anomalies
        random_state=42,
    )
    model.fit(X)

    result_df = supplier_df.copy()
    result_df["anomaly_score"] = model.decision_function(X).tolist()
    result_df["is_anomaly"] = (model.predict(X) == -1).tolist()

    anomalies = result_df[result_df["is_anomaly"]]
    print(
        f"[Isolation Forest] Monitored: {len(result_df)} suppliers | "
        f"Anomalies: {len(anomalies)} ({100*len(anomalies)/len(result_df):.1f}%)"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return model, result_df


def detect_anomaly(model, supplier_metrics: dict) -> dict:
    """
    Real-time single-supplier anomaly check.
    Returns: {is_anomaly, score}
    """
    X = np.array([[supplier_metrics.get(f, 0) for f in SUPPLIER_FEATURES]])
    score = float(model.decision_function(X)[0])
    flag = bool(model.predict(X)[0] == -1)
    return {"is_anomaly": flag, "score": round(score, 4)}


def get_all_supplier_scores(model, supplier_df) -> list:
    """Return anomaly scores for all suppliers as a list of dicts."""
    X = supplier_df[SUPPLIER_FEATURES].fillna(0)
    scores = model.decision_function(X).tolist()
    flags = (model.predict(X) == -1).tolist()

    results = []
    for i, row in supplier_df.iterrows():
        results.append(
            {
                "department": row["department"],
                "avg_delivery_time": round(float(row["avg_delivery_time"]), 2),
                "price_deviation": round(float(row["price_deviation"]), 4),
                "fulfillment_rate": round(float(row["fulfillment_rate"]), 3),
                "complaint_freq": int(row["complaint_freq"]),
                "anomaly_score": round(scores[list(supplier_df.index).index(i)], 4),
                "is_anomaly": flags[list(supplier_df.index).index(i)],
            }
        )
    return results
