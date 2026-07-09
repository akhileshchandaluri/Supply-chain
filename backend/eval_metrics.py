import pandas as pd
import joblib
from src.feature_engineering import build_risk_features, build_demand_features
from src.rf_risk import evaluate_random_forest
from src.xgboost_demand import evaluate_xgboost

print("Loading dataset...")
df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")

# Clean column names for feature pipelines
rename_map = {
    "Days for shipping (real)": "actual_days",
    "Days for shipment (scheduled)": "scheduled_days",
    "Order Item Discount Rate": "discount_rate",
    "Order Item Product Price": "order_value",
    "Department Name": "department",
    "Late_delivery_risk": "late_risk",
    "Shipping Mode": "shipping_mode",
    "order date (DateOrders)": "order_date",
    "Category Name": "category",
    "Customer Segment": "customer_segment",
    "Order Country": "country",
    "Order Item Quantity": "quantity"
}
df = df.rename(columns=rename_map)
df["order_date"] = pd.to_datetime(df["order_date"]).dt.normalize()

print("Building features...")
rf_df = build_risk_features(df)
daily_df = build_demand_features(df)

print("\n--- 1. Evaluating Random Forest (Risk Classification) ---")
rf_model = joblib.load("models/rf_risk.pkl")
rf_metrics = evaluate_random_forest(rf_model, rf_df)
print(f"Accuracy:  {rf_metrics['accuracy']:.4f}")
print(f"Precision: {rf_metrics['precision']:.4f}")
print(f"Recall:    {rf_metrics['recall']:.4f}")
print(f"F1 Score:  {rf_metrics['f1']:.4f}")

print("\n--- 2. Evaluating XGBoost (Demand Forecasting) ---")
xgb_model = joblib.load("models/xgboost_demand.pkl")
xgb_metrics = evaluate_xgboost(xgb_model, daily_df, horizon=1)

print(f"RMSE: {xgb_metrics['RMSE']:.2f}")
print(f"MAE:  {xgb_metrics['MAE']:.2f}")
print(f"R²:   {xgb_metrics['R2']:.4f}")
print(f"WAPE: {xgb_metrics['WAPE']:.2f}%")
