import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

FEATURE_COLS = [
    "day_of_week",
    "month",
    "quarter",
    "day_of_year",
    "is_weekend",
    "rolling_7d_mean",
    "rolling_30d_mean",
    "rolling_7d_std",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
]


def train_xgboost(daily_df, horizon=1, save_path="models/xgboost_demand.pkl"):
    """
    Train XGBoost for demand forecasting.
    horizon: 1, 3, or 7 days ahead
    Uses time-based 80/20 split — NO shuffle to avoid data leakage.
    """
    X = daily_df[FEATURE_COLS]
    y = daily_df["demand"].shift(-horizon).dropna()
    X = X.iloc[: len(y)]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    # Use WAPE instead of MAPE to avoid division by zero when demand is 0
    y_test_sum = np.sum(np.abs(y_test))
    wape = float(np.sum(np.abs(y_test - y_pred)) / y_test_sum) * 100 if y_test_sum != 0 else 0.0

    print(
        f"[XGBoost | horizon={horizon}d] RMSE={rmse:.2f}  "
        f"MAE={mae:.2f}  R²={r2:.3f}  WAPE={wape:.1f}%"
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return model, {"RMSE": rmse, "MAE": mae, "R2": r2, "WAPE": wape}


def evaluate_xgboost(model, daily_df, horizon=1):
    """Evaluate a loaded demand model using the same time split as training."""
    X = daily_df[FEATURE_COLS]
    y = daily_df["demand"].shift(-horizon).dropna()
    X = X.iloc[: len(y)]

    split = int(len(X) * 0.8)
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    if X_test.empty:
        return {}

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    y_test_sum = np.sum(np.abs(y_test))
    wape = float(np.sum(np.abs(y_test - y_pred)) / y_test_sum) * 100 if y_test_sum != 0 else 0.0

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "WAPE": wape}


def _make_feature_row(history: pd.Series, date: pd.Timestamp) -> dict:
    return {
        "day_of_week": date.dayofweek,
        "month": date.month,
        "quarter": date.quarter,
        "day_of_year": date.dayofyear,
        "is_weekend": int(date.dayofweek >= 5),
        "rolling_7d_mean": float(history.tail(7).mean()),
        "rolling_30d_mean": float(history.tail(30).mean()),
        "rolling_7d_std": float(history.tail(7).std() if len(history.tail(7)) > 1 else 0.0),
        "lag_1": float(history.iloc[-1]),
        "lag_7": float(history.iloc[-7] if len(history) >= 7 else history.iloc[-1]),
        "lag_14": float(history.iloc[-14] if len(history) >= 14 else history.iloc[-1]),
        "lag_30": float(history.iloc[-30] if len(history) >= 30 else history.iloc[-1]),
    }


def predict_demand(model, daily_df, horizon=7):
    """Return recursive demand forecast for the next `horizon` calendar days."""
    history = daily_df["demand"].astype(float).copy()
    last_date = daily_df.index[-1]
    forecasts = []

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(days=step + 1)
        row = pd.DataFrame([_make_feature_row(history, next_date)])[FEATURE_COLS]
        pred = max(0.0, float(model.predict(row)[0]))
        forecasts.append(pred)
        history.loc[next_date] = pred

    return forecasts
