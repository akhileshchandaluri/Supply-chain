"""
Sweep Custom-Order form combinations through the real pipeline and find one
input set per RL action, so all 6 outputs can be demonstrated in the UI.

Run:  backend/venv/Scripts/python.exe find_rl_testcases.py
"""
import sys, os, itertools, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib, numpy as np
from data_preprocessing import load_data
from feature_engineering import build_demand_features
from integration import run_pipeline

# ─── Presets mirrored from Dashboard.jsx (must stay in sync) ──────────────────
URGENCY = {
    "Standard":  {"days_to_delivery": 6, "shipping_mode_enc": 0},
    "Expedited": {"days_to_delivery": 3, "shipping_mode_enc": 1},
    "Emergency": {"days_to_delivery": 1, "shipping_mode_enc": 2},
}
RELIABILITY = {
    "High":   {"fulfillment_rate": 0.95, "avg_delivery_time": 3.0, "complaint_freq": 3.0,  "price_deviation": 0.05, "supplier_delay_rate": 0.10},
    "Medium": {"fulfillment_rate": 0.85, "avg_delivery_time": 5.0, "complaint_freq": 10.0, "price_deviation": 0.10, "supplier_delay_rate": 0.20},
    "Low":    {"fulfillment_rate": 0.55, "avg_delivery_time": 9.0, "complaint_freq": 28.0, "price_deviation": 0.40, "supplier_delay_rate": 0.45},
}
SOURCE = {"Mumbai": 0, "Delhi": 1, "Chennai": 2, "Kochi": 11, "Guwahati": 17}
HUB = {"Bengaluru": 6, "Mumbai": 0, "Chennai": 2, "Delhi": 1}


def build_payload(quantity, urgency, hub, inventory, reliability, source, order_value, discount_rate):
    preset = URGENCY[urgency]
    rel = RELIABILITY[reliability]
    goal = HUB[hub]
    start = SOURCE[source]
    if start == goal:
        start = 1 if goal == 0 else 0
    value = order_value if order_value > 0 else max(1, quantity) * 100.0
    return {
        "inventory": float(max(0, inventory)),
        "days_to_delivery": float(preset["days_to_delivery"]),
        "start_node": start,
        "goal_node": goal,
        "order_quantity": float(max(1, quantity)),
        "risk_features": {
            "shipping_mode_enc": preset["shipping_mode_enc"],
            "actual_days": preset["days_to_delivery"] + 1,
            "scheduled_days": preset["days_to_delivery"],
            "discount_rate": min(1, max(0, discount_rate)),
            "order_value": value,
            "supplier_delay_rate": rel["supplier_delay_rate"],
            "days_buffer": 0.0,
            "delay_gap": 1.0,
        },
        "supplier_metrics": {
            "avg_delivery_time": rel["avg_delivery_time"],
            "price_deviation": rel["price_deviation"],
            "fulfillment_rate": rel["fulfillment_rate"],
            "complaint_freq": rel["complaint_freq"],
        },
    }


def main():
    df = load_data("data/DataCoSupplyChainDataset.csv")
    daily = build_demand_features(df)
    xgb = joblib.load("models/xgboost_demand.pkl")
    rf = joblib.load("models/rf_risk.pkl")
    iso = joblib.load("models/isolation_forest.pkl")
    q = np.load("models/q_table.npy")

    # Sweep grid over the controllable form fields.
    quantities = [20, 200, 500, 900, 1500, 2500]
    inventories = [5, 40, 150, 400, 800, 1500]
    urgencies = list(URGENCY)
    reliabilities = list(RELIABILITY)
    order_values = [100, 500, 2000]
    discounts = [0.1, 0.5]

    found = {}          # action -> form settings
    action_counts = {}
    for qty, inv, urg, rel, ov, dr in itertools.product(
        quantities, inventories, urgencies, reliabilities, order_values, discounts
    ):
        form = dict(quantity=qty, urgency=urg, hub="Bengaluru", inventory=inv,
                    reliability=rel, source="Mumbai", order_value=ov, discount_rate=dr)
        payload = build_payload(**form)
        current_state = {
            "inventory": payload["inventory"],
            "days_to_delivery": payload["days_to_delivery"],
            "start_node": payload["start_node"],
            "goal_node": payload["goal_node"],
            "order_quantity": payload["order_quantity"],
            "risk_features": payload["risk_features"],
            "supplier_metrics": payload["supplier_metrics"],
        }
        res = run_pipeline(xgb, rf, iso, q, daily, current_state)
        action = res["rl_action"]["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
        if action not in found:
            found[action] = dict(
                form=form,
                demand=res["demand_7d_avg"],
                risk=res["risk_level"],
                anomaly=res["anomaly"]["is_anomaly"],
            )

    print("\n=== ACTION FREQUENCY across sweep ===")
    for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {a:20s} {c}")

    all_actions = ["HOLD", "REORDER_SMALL", "REORDER_MEDIUM", "REORDER_LARGE",
                   "EMERGENCY_REORDER", "SWITCH_SUPPLIER"]
    print("\n=== ONE FORM SETUP PER ACTION ===")
    for a in all_actions:
        if a in found:
            f = found[a]["form"]
            print(f"\n[{a}]  (demand={found[a]['demand']}, risk={found[a]['risk']}, anomaly={found[a]['anomaly']})")
            print(f"    Target Quantity : {f['quantity']}")
            print(f"    Delivery Urgency: {f['urgency']}")
            print(f"    Destination Hub : {f['hub']}")
            print(f"    Current Inventory: {f['inventory']}")
            print(f"    Supplier Reliab.: {f['reliability']}")
            print(f"    Source Warehouse: {f['source']}")
            print(f"    Order Value     : {f['order_value']}")
            print(f"    Discount Rate   : {f['discount_rate']}")
        else:
            print(f"\n[{a}]  *** NOT REACHABLE with any swept combination ***")


if __name__ == "__main__":
    main()
