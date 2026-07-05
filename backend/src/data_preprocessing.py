import pandas as pd
import numpy as np


def load_data(path="data/DataCoSupplyChainDataset.csv"):
    df = pd.read_csv(path, encoding="latin-1")

    # Detect date column name (varies between dataset versions)
    order_date_col = next(
        (c for c in df.columns if "order date" in c.lower()), None
    )
    shipping_date_col = next(
        (c for c in df.columns if "shipping date" in c.lower()), None
    )

    if order_date_col is None:
        raise ValueError(f"Cannot find order date column. Columns: {list(df.columns)}")

    df["order_date"] = pd.to_datetime(df[order_date_col], format="%m/%d/%Y %H:%M", errors="coerce")
    if shipping_date_col:
        df["shipping_date"] = pd.to_datetime(df[shipping_date_col], format="%m/%d/%Y %H:%M", errors="coerce")

    # Category column name also varies
    category_col = next(
        (c for c in df.columns if "category name" in c.lower()), "Category Name"
    )

    # Rename for convenience — handle both dataset variants
    rename_map = {
        "Order Item Quantity":            "quantity",
        "Late_delivery_risk":             "late_risk",
        "Days for shipping (real)":       "actual_days",
        "Days for shipment (scheduled)":  "scheduled_days",
        "Shipping Mode":                  "shipping_mode",
        "Order Item Discount Rate":       "discount_rate",
        category_col:                     "category",
        "Customer Segment":               "customer_segment",
        "Order Item Total":               "order_value",
        "Department Name":                "department",
        "Order Country":                  "country",
        "Order City":                     "city",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Drop duplicates and nulls
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["quantity", "order_date"], inplace=True)

    return df


def basic_eda(df):
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"\nLate risk distribution:\n{df['late_risk'].value_counts()}")
    print(f"\nShipping modes:\n{df['shipping_mode'].value_counts()}")
    print(f"\nNull values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df.describe()
