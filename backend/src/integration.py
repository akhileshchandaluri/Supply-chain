"""
integration.py — SmartChain AI Master Pipeline Orchestrator

Chains all five AI components in sequence:
  XGBoost → Random Forest → Isolation Forest → Q-Learning RL → Routing

Design principle: models are injected as parameters (already loaded in memory
by main.py at startup) rather than re-loaded from disk on every call.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from xgboost_demand import predict_demand
from rf_risk import predict_risk
from isolation_forest import detect_anomaly
from rl_agent import get_action
from astar_routing import astar
from xai_explainer import generate_pipeline_explanation


# Risk label → numeric score used as RL input
_RISK_SCORE_MAP = {"LOW": 0.2, "MEDIUM": 0.55, "HIGH": 0.85}


def run_pipeline(
    xgb_model,
    rf_model,
    iso_model,
    q_table: np.ndarray,
    daily_df,
    current_state: dict,
) -> dict:
    """
    Executes the complete SmartChain AI Orchestration Pipeline.
    
    Workflow:
    1. XGBoost Forecasts global demand for the next 7 days.
    2. Random Forest Classifies the real-time logistical delivery risk (Low/Med/High).
    3. Isolation Forest Detects supplier quality-control anomalies.
    4. Q-Learning Agent ingests the Forecast, Risk, and Anomaly states to determine the optimal strategic action.
    5. A* Algorithm calculates the physical logistics route based on the Agent's decision.
    """

    # ── Step 1: XGBoost — 7-day demand forecast ───────────────────────────────
    base_forecast = predict_demand(xgb_model, daily_df, horizon=7)
    risk_features = current_state.get("risk_features", {})
    order_qty = float(current_state.get("order_quantity", risk_features.get("order_quantity", 1.0)))
    order_value = float(risk_features.get("order_value", 0.0))
    avg_value_per_unit = float(current_state.get("avg_order_value_per_unit", 100.0))

    # Per-order scaling based on order characteristics.
    # Higher per-unit value suggests higher regional demand.
    value_ratio = order_value / max(avg_value_per_unit, 1.0)
    value_scale = min(1.8, max(0.5, value_ratio))

    # Urgency adds pressure — rush orders hint at demand spikes
    days_to_del = current_state.get("days_to_delivery", 5)
    urgency_scale = 1.0 + max(0, (3.0 - days_to_del)) * 0.12

    # Order SIZE drives demand: a larger Target Quantity implies a bigger demand
    # spike, which is what pushes the RL agent toward larger / emergency reorders.
    # (Real dataset orders have qty ~1-5, so this barely affects the simulate path.)
    quantity_scale = min(8.0, 1.0 + order_qty / 400.0)

    # Combine into a per-order multiplier (centered around 1.0)
    order_multiplier = value_scale * urgency_scale * quantity_scale

    # Add order-specific noise using the order value as seed for reproducibility
    # This ensures different orders get different forecasts
    rng = np.random.RandomState(int(abs(order_value * 100 + order_qty * 17 + days_to_del * 31)) % (2**31))
    noise = rng.normal(0, 0.08, len(base_forecast))

    forecast = [
        max(0.0, float(v) * order_multiplier * (1.0 + noise[i]))
        for i, v in enumerate(base_forecast)
    ]
    demand_7d = float(np.mean(forecast))

    # ── Step 2: Random Forest — risk classification ───────────────────────────
    risk_result   = predict_risk(rf_model, risk_features)
    risk_label    = risk_result["label"]                           # LOW/MEDIUM/HIGH
    risk_score    = _RISK_SCORE_MAP.get(risk_label, 0.5)          # numeric for RL
    risk_proba    = risk_result.get("probabilities", [])

    # ── Step 3: Isolation Forest — supplier anomaly detection ─────────────────
    supplier_metrics = current_state.get("supplier_metrics", {
        "avg_delivery_time": 5.0,
        "price_deviation":   0.1,
        "fulfillment_rate":  0.85,
        "complaint_freq":    10.0,
    })
    anomaly_result = detect_anomaly(iso_model, supplier_metrics)
    anomaly_flag   = int(anomaly_result["is_anomaly"])

    # ── Step 4: Q-Learning RL — inventory action decision ─────────────────────
    rl_result = get_action(
        q_table,
        inv     = current_state.get("inventory", 100),
        demand  = demand_7d,                   # use live XGBoost forecast
        risk    = risk_score,                  # use RF-derived numeric risk
        anomaly = anomaly_flag,                # use Isolation Forest flag
        days    = current_state.get("days_to_delivery", 5),
    )

    # ── Step 4.5: Optimization Layer — physical warehouse allocation ──────────
    # Removed GLOP optimization to simplify pipeline architecture.
    optimization = None

    # ── Step 5: Routing — always A* (heuristic shortest-path) ────────────────
    start = int(current_state.get("start_node", 0))
    goal  = int(current_state.get("goal_node",  6))

    # Emergency conditions: RL says so, OR risk is HIGH, OR anomaly detected.
    # (Kept for the route's context label/trigger — the algorithm is A* either way.)
    is_emergency = (
        rl_result["action"] == "EMERGENCY_REORDER"
        or risk_score > 0.8
        or anomaly_flag == 1
    )

    # Routing is wrapped so a graph/algorithm failure cannot crash the pipeline;
    # the fallback keeps the same route dict shape the frontend + summary expect.
    try:
        route = astar(start, goal)
        if is_emergency:
            route["type"] = "EMERGENCY (A*)"
            if rl_result["action"] == "EMERGENCY_REORDER":
                route["trigger"] = "RL agent: EMERGENCY_REORDER"
            elif anomaly_flag:
                route["trigger"] = "Isolation Forest: supplier anomaly detected"
            else:
                route["trigger"] = f"Random Forest: risk score {risk_score} > 0.8"
        else:
            route["type"]    = "STANDARD (A*)"
            route["trigger"] = "Normal operations - A* shortest-path routing"
    except Exception as e:
        print(f"[Pipeline] Routing step failed: {e}")
        route = {
            "path": [],
            "path_names": [],
            "total_cost": 0,
            "nodes_explored": 0,
            "algorithm": "A*",
            "type": "EMERGENCY (A*)" if is_emergency else "STANDARD (A*)",
            "trigger": "Routing unavailable — using fallback",
            "error": str(e),
        }

    # ── Assemble unified response ─────────────────────────────────────────────
    pipeline_data = {
        # Demand
        "demand_forecast_7d": [round(v, 2) for v in forecast],
        "demand_7d_avg":      round(demand_7d, 2),

        # Risk
        "risk_level":         risk_label,
        "risk_score":         risk_score,
        "risk_probabilities": {
            "LOW":    round(risk_proba[0], 4) if len(risk_proba) > 0 else None,
            "MEDIUM": round(risk_proba[1], 4) if len(risk_proba) > 1 else None,
            "HIGH":   round(risk_proba[2], 4) if len(risk_proba) > 2 else None,
        },

        # Anomaly
        "anomaly": {
            "is_anomaly":   bool(anomaly_result["is_anomaly"]),
            "score":        anomaly_result.get("score"),
        },

        # RL
        "rl_action": rl_result,


        # Routing
        "route":        route,
        "is_emergency": is_emergency,

        "pipeline_summary": {
            "step_1_demand":  f"XGBoost 7-day avg: {demand_7d:.1f} units",
            "step_2_risk":    f"Random Forest: {risk_label} (score {risk_score})",
            "step_3_anomaly": f"Isolation Forest: {'ANOMALY' if anomaly_flag else 'Normal'} (score {anomaly_result.get('score')})",
            "step_4_rl":      f"Q-Learning: {rl_result['action']}",
            "step_5_route":   f"{route['type']}: {' → '.join(route.get('path_names', []))}",
        },

        # System State
        "current_state": current_state,
    }

    # ── Layer 6: XAI — plain-language executive audit of the full decision ─────
    # Best-effort: returns "Audit generation unavailable" on missing key/timeout.
    pipeline_data["ai_explanation"] = generate_pipeline_explanation(pipeline_data)

    return pipeline_data
