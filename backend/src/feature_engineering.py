import pandas as pd
import numpy as np


def build_demand_features(df):
    """
    Aggregate daily demand, then add temporal + lag features for XGBoost.
    """
    # Daily aggregation
    daily = (
        df.groupby("order_date")
        .agg(demand=("quantity", "sum"))
        .reset_index()
        .sort_values("order_date")
    )

    daily.set_index("order_date", inplace=True)
    daily = daily.asfreq("D", fill_value=0)  # fill missing days with 0

    # Temporal features
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["quarter"] = daily.index.quarter
    daily["day_of_year"] = daily.index.dayofyear
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

    # Rolling statistics
    daily["rolling_7d_mean"] = daily["demand"].shift(1).rolling(7).mean()
    daily["rolling_30d_mean"] = daily["demand"].shift(1).rolling(30).mean()
    daily["rolling_7d_std"] = daily["demand"].shift(1).rolling(7).std()

    # Lag features
    for lag in [1, 7, 14, 30]:
        daily[f"lag_{lag}"] = daily["demand"].shift(lag)

    daily.dropna(inplace=True)
    return daily


def build_risk_features(df):
    """
    Build features for Random Forest risk classification.
    Converts binary late_risk → 3-class: LOW / MEDIUM / HIGH.
    """
    df = df.copy()
    df["delay_gap"] = df["actual_days"] - df["scheduled_days"]

    def classify_risk(row):
        if row["delay_gap"] <= 0:
            return 0  # LOW
        elif row["delay_gap"] <= 2:
            return 1  # MEDIUM
        else:
            return 2  # HIGH

    df["risk_class"] = df.apply(classify_risk, axis=1)

    mode_map = {
        "Standard Class": 0,
        "Second Class": 1,
        "First Class": 2,
        "Same Day": 3,
    }
    df["shipping_mode_enc"] = df["shipping_mode"].map(mode_map).fillna(0)

    dept_delay = (
        df.groupby("department")["late_risk"]
        .mean()
        .rename("supplier_delay_rate")
    )
    df = df.join(dept_delay, on="department")

    df["days_buffer"] = df["scheduled_days"] - df["actual_days"].clip(lower=0)

    feature_cols = [
        "shipping_mode_enc",
        "actual_days",
        "scheduled_days",
        "discount_rate",
        "order_value",
        "supplier_delay_rate",
        "days_buffer",
        "delay_gap",
    ]

    rf_df = df[feature_cols + ["risk_class"]].dropna()
    return rf_df


def build_supplier_features(df):
    """
    Per-supplier (department-level) features for Isolation Forest.
    """
    supplier_stats = (
        df.groupby("department")
        .agg(
            avg_delivery_time=("actual_days", "mean"),
            price_deviation=(
                "order_value",
                lambda x: x.std() / (x.mean() + 1e-9),
            ),
            fulfillment_rate=("late_risk", lambda x: 1 - x.mean()),
            complaint_freq=("late_risk", "sum"),
        )
        .reset_index()
    )
    supplier_stats.fillna(0, inplace=True)
    return supplier_stats
