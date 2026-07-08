import sys
import os
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_preprocessing import load_data
from feature_engineering import build_demand_features, build_risk_features, build_supplier_features
from xgboost_demand import evaluate_xgboost
from rf_risk import evaluate_random_forest

DATA_PATH = "data/DataCoSupplyChainDataset.csv"
MODELS_DIR = "models"

print("Loading data...")
df = load_data(DATA_PATH)
daily_df = build_demand_features(df)
rf_df = build_risk_features(df)

print("\n--- Demand Forecasting (XGBoost) ---")
try:
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_demand.pkl"))
    metrics = evaluate_xgboost(xgb_model, daily_df)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
except Exception as e:
    print("Error evaluating XGBoost:", e)

print("\n--- Risk Classification (Random Forest) ---")
try:
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_risk.pkl"))
    metrics = evaluate_random_forest(rf_model, rf_df)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
except Exception as e:
    print("Error evaluating Random Forest:", e)

print("\n--- Isolation Forest (Anomaly Detection) ---")
try:
    iso_model = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    print("Anomaly Detection trained successfully. Output metrics depend on operational distribution.")
except Exception as e:
    print("Error loading Isolation Forest:", e)

print("\n--- Q-Learning Agent (RL) ---")
try:
    q_table = np.load(os.path.join(MODELS_DIR, "q_table.npy"))
    print(f"Q-table shape: {q_table.shape}")
    print("RL Agent trained successfully.")
except Exception as e:
    print("Error loading Q-table:", e)
