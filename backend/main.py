"""
SmartChain AI — FastAPI Backend
"""

import sys
import os

# Fix Windows charmap encoding for Unicode characters in print output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import joblib
import pandas as pd

# ─── Local imports ────────────────────────────────────────────────────────────
from data_preprocessing import load_data
from feature_engineering import (
    build_demand_features,
    build_risk_features,
    build_supplier_features,
)
from xgboost_demand import train_xgboost, predict_demand
from xgboost_demand import evaluate_xgboost
from rf_risk import train_random_forest, predict_risk, evaluate_random_forest
from isolation_forest import (
    train_isolation_forest,
    detect_anomaly,
    get_all_supplier_scores,
)
from graph_construction import get_graph_data, NODES
from astar_routing import astar
from dijkstra_routing import dijkstra
from rl_agent import train_agent, get_action
from integration import run_pipeline

# ─── Global State (defined before lifespan so it can reference these) ─────────
MODELS_DIR = "models"
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data/DataCoSupplyChainDataset.csv")

# The exact four files that must ALL exist for the system to be fully ready.
# If any one is missing at startup, training is triggered automatically.
REQUIRED_MODEL_FILES = {
    "xgboost_demand.pkl": "XGBoost demand forecaster",
    "rf_risk.pkl":        "Random Forest risk classifier",
    "isolation_forest.pkl": "Isolation Forest anomaly detector",
    "q_table.npy":        "Q-Learning RL agent",
}

state = {
    "models_loaded": False,
    "training_in_progress": False,
    "xgb_metrics": {},
    "rf_metrics": {},
    "train_error": None,
    "daily_df": None,
    "supplier_df": None,
}

xgb_model         = None
rf_model          = None
iso_model         = None
q_table           = None
daily_df_cache    = None
supplier_df_cache = None
raw_df_cache      = None


def _all_model_files_present() -> bool:
    """
    Strict check: returns True only when ALL four required model files exist
    on disk. Logs exactly which file(s) are missing so the startup message
    is unambiguous.
    """
    missing = [
        filename
        for filename in REQUIRED_MODEL_FILES
        if not os.path.exists(os.path.join(MODELS_DIR, filename))
    ]
    if missing:
        for filename in missing:
            label = REQUIRED_MODEL_FILES[filename]
            print(f"[Startup] Missing model file: {filename}  ({label})")
        return False
    return True


# ─── Lifespan: startup + strict auto-train trigger ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load whichever model files already exist into memory.
    _load_models_if_exist()
    # Pre-build feature DataFrames from the CSV (used by forecast + anomaly endpoints).
    _load_data_cache()

    # Strict check: trigger training if ANY of the four required files is absent.
    if not _all_model_files_present():
        if os.path.exists(DATA_PATH):
            if not state["training_in_progress"]:
                print("[Startup] One or more model files missing — "
                      "starting background training automatically...")
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, _do_training, 10000)
        else:
            print(f"[Startup] WARNING: Dataset not found at '{DATA_PATH}'. "
                  "Cannot auto-train. Place DataCoSupplyChainDataset.csv in backend/data/.")
    else:
        print("[Startup] All 4 model files verified — skipping training.")

    yield


# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartChain AI API",
    description="Intelligent Supply Chain Management",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_models_if_exist():
    """
    Load each model file into memory if it exists on disk.
    Sets state['models_loaded'] = True only when ALL FOUR models are loaded.
    Does NOT trigger training — that is the lifespan's responsibility via
    _all_model_files_present().
    """
    global xgb_model, rf_model, iso_model, q_table
    try:
        xgb_path = os.path.join(MODELS_DIR, "xgboost_demand.pkl")
        rf_path  = os.path.join(MODELS_DIR, "rf_risk.pkl")
        iso_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
        rl_path  = os.path.join(MODELS_DIR, "q_table.npy")

        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            print(f"[Startup] Loaded: xgboost_demand.pkl")
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
            print(f"[Startup] Loaded: rf_risk.pkl")
        if os.path.exists(iso_path):
            iso_model = joblib.load(iso_path)
            print(f"[Startup] Loaded: isolation_forest.pkl")
        if os.path.exists(rl_path):
            q_table = np.load(rl_path)
            print(f"[Startup] Loaded: q_table.npy")

        # Only mark fully ready when all four are in memory.
        if all(m is not None for m in [xgb_model, rf_model, iso_model, q_table]):
            state["models_loaded"] = True
            print("[Startup] All 4 models loaded into memory.")
    except Exception as e:
        print(f"[Startup] Error loading saved models: {e}")


def _load_data_cache():
    global daily_df_cache, supplier_df_cache, raw_df_cache
    if os.path.exists(DATA_PATH):
        try:
            df = load_data(DATA_PATH)
            raw_df_cache = df
            daily_df_cache = build_demand_features(df)
            supplier_df_cache = build_supplier_features(df)
            _refresh_loaded_model_metrics()
        except Exception as e:
            print(f"[Startup] Could not load data: {e}")


def _refresh_loaded_model_metrics():
    """Populate metrics for saved models so the UI is not blank after startup."""
    if daily_df_cache is not None and xgb_model is not None and not state["xgb_metrics"]:
        try:
            state["xgb_metrics"] = evaluate_xgboost(xgb_model, daily_df_cache)
        except Exception as e:
            print(f"[Startup] Could not evaluate XGBoost model: {e}")

    if raw_df_cache is not None and rf_model is not None and not state["rf_metrics"]:
        try:
            rf_df = build_risk_features(raw_df_cache)
            state["rf_metrics"] = evaluate_random_forest(rf_model, rf_df)
        except Exception as e:
            print(f"[Startup] Could not evaluate Random Forest model: {e}")


# Startup is handled by the lifespan context manager above


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class RiskFeatures(BaseModel):
    shipping_mode_enc: float = 0
    actual_days: float = 5
    scheduled_days: float = 4
    discount_rate: float = 0.1
    order_value: float = 500
    supplier_delay_rate: float = 0.2
    days_buffer: float = 0
    delay_gap: float = 0


class RouteRequest(BaseModel):
    start: int = 0
    goal: int = 6


class RLRequest(BaseModel):
    inventory: float = 120
    demand: float = 80
    risk: float = 0.4
    anomaly: int = 0
    days: float = 5


class SupplierMetrics(BaseModel):
    avg_delivery_time: float = 5.0
    price_deviation: float = 0.1
    fulfillment_rate: float = 0.85
    complaint_freq: float = 10.0


class TrainRequest(BaseModel):
    episodes: Optional[int] = 10000


# ─── Pipeline request schema ──────────────────────────────────────────────────
class RiskFeaturesInline(BaseModel):
    """8 Random Forest input features (same as RiskFeatures above)."""
    shipping_mode_enc:  float = 0.0
    actual_days:        float = 5.0
    scheduled_days:     float = 4.0
    discount_rate:      float = 0.10
    order_value:        float = 500.0
    supplier_delay_rate:float = 0.20
    days_buffer:        float = 0.0
    delay_gap:          float = 0.0


class SupplierMetricsInline(BaseModel):
    """4 Isolation Forest input features."""
    avg_delivery_time: float = 5.0
    price_deviation:   float = 0.1
    fulfillment_rate:  float = 0.85
    complaint_freq:    float = 10.0


class PipelineRequest(BaseModel):
    """
    Full environment state for the master pipeline run.

    Required fields
    ---------------
    inventory        : Current inventory level (units)
    days_to_delivery : Days until next scheduled delivery
    start_node       : Source node ID  (0–17, see /api/nodes)
    goal_node        : Destination node ID (0–17)
    risk_features    : 8-feature dict for Random Forest
    supplier_metrics : 4-feature dict for Isolation Forest
    """
    inventory:         float                = 120.0
    days_to_delivery:  float                = 5.0
    start_node:        int                  = 0
    goal_node:         int                  = 6
    risk_features:     RiskFeaturesInline   = RiskFeaturesInline()
    supplier_metrics:  SupplierMetricsInline = SupplierMetricsInline()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    return {
        "status": "running",
        "models_loaded": state["models_loaded"],
        "training_in_progress": state["training_in_progress"],
        "data_available": daily_df_cache is not None,
        "xgb_metrics": state["xgb_metrics"],
        "rf_metrics": state["rf_metrics"],
        "train_error": state["train_error"],
    }


def _do_training(episodes: int):
    global xgb_model, rf_model, iso_model, q_table, daily_df_cache, supplier_df_cache, raw_df_cache

    state["training_in_progress"] = True
    state["train_error"] = None
    try:
        print("[Train] Loading data...")
        df = load_data(DATA_PATH)

        print("[Train] Feature engineering...")
        daily_df     = build_demand_features(df)
        rf_df        = build_risk_features(df)
        supplier_df  = build_supplier_features(df)

        raw_df_cache      = df
        daily_df_cache    = daily_df
        supplier_df_cache = supplier_df

        print("[Train] Training XGBoost...")
        xgb_model, xgb_metrics = train_xgboost(daily_df, save_path=f"{MODELS_DIR}/xgboost_demand.pkl")
        state["xgb_metrics"] = xgb_metrics

        print("[Train] Training Random Forest...")
        rf_model, rf_metrics = train_random_forest(rf_df, save_path=f"{MODELS_DIR}/rf_risk.pkl")
        state["rf_metrics"] = rf_metrics

        print("[Train] Training Isolation Forest...")
        iso_model, _ = train_isolation_forest(supplier_df, save_path=f"{MODELS_DIR}/isolation_forest.pkl")

        print(f"[Train] Training Q-Learning ({episodes} episodes)...")
        agent, _ = train_agent(episodes=episodes, save_path=f"{MODELS_DIR}/q_table.npy")
        q_table = agent.q_table

        state["models_loaded"] = True
        print("[Train] All models trained successfully!")
    except Exception as e:
        state["train_error"] = str(e)
        print(f"[Train] ERROR: {e}")
    finally:
        state["training_in_progress"] = False


@app.post("/api/train")
async def train_models(req: TrainRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found at '{DATA_PATH}'. "
                   "Please place DataCoSupplyChainDataset.csv in the backend/data/ folder.",
        )
    if state["training_in_progress"]:
        raise HTTPException(status_code=409, detail="Training already in progress.")

    background_tasks.add_task(_do_training, req.episodes)
    return {"message": "Training started in background. Poll /api/status for updates."}


@app.get("/api/graph")
def get_graph():
    return get_graph_data()


@app.get("/api/demand/forecast")
def forecast_demand(horizon: int = 7):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="XGBoost model not trained yet.")
    if daily_df_cache is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    forecasts = predict_demand(xgb_model, daily_df_cache, horizon=horizon)

    # Return last 30 days of actual data too
    actual = daily_df_cache["demand"].tail(30).tolist()
    dates  = [str(d.date()) for d in daily_df_cache.index[-30:]]

    last_date = daily_df_cache.index[-1]
    future_dates = [
        str((last_date + pd.Timedelta(days=i + 1)).date())
        for i in range(horizon)
    ]

    return {
        "actual_dates": dates,
        "actual_demand": actual,
        "forecast_dates": future_dates,
        "forecast_demand": forecasts,
        "metrics": state["xgb_metrics"],
    }


@app.post("/api/risk/predict")
def predict_risk_endpoint(features: RiskFeatures):
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Random Forest model not trained yet.")
    result = predict_risk(rf_model, features.dict())
    result["feature_importances"] = dict(
        zip(
            ["shipping_mode_enc", "actual_days", "scheduled_days",
             "discount_rate", "order_value", "supplier_delay_rate",
             "days_buffer", "delay_gap"],
            rf_model.feature_importances_.tolist(),
        )
    )
    return result


@app.get("/api/anomaly/detect")
def detect_anomalies():
    if iso_model is None:
        raise HTTPException(status_code=503, detail="Isolation Forest model not trained yet.")
    if supplier_df_cache is None:
        raise HTTPException(status_code=503, detail="Supplier data not available.")

    results = get_all_supplier_scores(iso_model, supplier_df_cache)
    total    = len(results)
    anomaly_count = sum(1 for r in results if r["is_anomaly"])
    return {
        "suppliers": results,
        "total": total,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(anomaly_count / total * 100, 1) if total > 0 else 0,
    }


@app.get("/api/supplier/scorecard")
def supplier_scorecard():
    """Return the global department-level supplier health baseline."""
    return detect_anomalies()


@app.post("/api/anomaly/check")
def check_supplier(metrics: SupplierMetrics):
    if iso_model is None:
        raise HTTPException(status_code=503, detail="Isolation Forest model not trained yet.")
    return detect_anomaly(iso_model, metrics.dict())


@app.post("/api/route/astar")
def astar_route(req: RouteRequest):
    if req.start not in NODES or req.goal not in NODES:
        raise HTTPException(status_code=400, detail="Invalid node IDs.")
    return astar(req.start, req.goal)


@app.post("/api/route/dijkstra")
def dijkstra_route(req: RouteRequest):
    if req.start not in NODES or req.goal not in NODES:
        raise HTTPException(status_code=400, detail="Invalid node IDs.")
    return dijkstra(req.start, req.goal)


@app.post("/api/rl/action")
def rl_action(req: RLRequest):
    if q_table is None:
        raise HTTPException(status_code=503, detail="Q-table not trained yet.")
    return get_action(q_table, req.inventory, req.demand, req.risk, req.anomaly, req.days)


@app.get("/api/nodes")
def get_nodes():
    return [
        {"id": nid, "name": data[0], "lat": data[1], "lon": data[2], "type": data[3]}
        for nid, data in NODES.items()
    ]


# ─── Master pipeline endpoint ────────────────────────────────────────────────
@app.post("/api/pipeline/run", summary="Run complete AI pipeline end-to-end")
def run_full_pipeline(req: PipelineRequest):
    """
    Execute all 5 AI stages in sequence and return a unified result:

    1. **XGBoost** — 7-day demand forecast
    2. **Random Forest** — risk classification (LOW / MEDIUM / HIGH)
    3. **Isolation Forest** — supplier anomaly detection
    4. **Q-Learning RL** — optimal inventory action
    5. **A* or Dijkstra** — route selection based on urgency

    The routing algorithm is chosen automatically:
    - A* (emergency speed) if RL says EMERGENCY_REORDER, risk is HIGH, or an anomaly is detected
    - Dijkstra (cost-optimal) otherwise
    """
    # Guard: all four models must be loaded
    if not state["models_loaded"]:
        missing = [
            name for name in REQUIRED_MODEL_FILES
            if not os.path.exists(os.path.join(MODELS_DIR, name))
        ]
        detail = "Models not yet trained."
        if missing:
            detail += f" Missing: {', '.join(missing)}."
        if state["training_in_progress"]:
            detail += " Training is currently running — please retry in a few minutes."
        raise HTTPException(status_code=503, detail=detail)

    if daily_df_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Feature data not loaded. Ensure the dataset CSV is present and restart.",
        )

    # Validate node IDs
    if req.start_node not in NODES:
        raise HTTPException(status_code=400, detail=f"start_node {req.start_node} is not a valid node ID.")
    if req.goal_node not in NODES:
        raise HTTPException(status_code=400, detail=f"goal_node {req.goal_node} is not a valid node ID.")
    if req.start_node == req.goal_node:
        raise HTTPException(status_code=400, detail="start_node and goal_node must be different.")

    # Build current_state dict that integration.py expects
    current_state = {
        "inventory":        req.inventory,
        "days_to_delivery": req.days_to_delivery,
        "start_node":       req.start_node,
        "goal_node":        req.goal_node,
        "risk_features":    req.risk_features.dict(),
        "supplier_metrics": req.supplier_metrics.dict(),
    }

    try:
        result = run_pipeline(
            xgb_model   = xgb_model,
            rf_model    = rf_model,
            iso_model   = iso_model,
            q_table     = q_table,
            daily_df    = daily_df_cache,
            current_state = current_state,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {e}")

    return result


@app.get("/api/simulate/order", summary="Simulate a live data stream order")
def simulate_live_order():
    """
    Simulate a live order by picking a random row from the dataset,
    extracting its exact features, and passing it through the master pipeline.
    """
    global raw_df_cache, supplier_df_cache
    if raw_df_cache is None:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=503, detail="Dataset not found.")
        raw_df_cache = load_data(DATA_PATH)
    if supplier_df_cache is None and raw_df_cache is not None:
        supplier_df_cache = build_supplier_features(raw_df_cache)
        
    df = raw_df_cache
    sample = df.sample(n=1).iloc[0]
    
    # Re-build risk features for this specific row
    mode_map = {"Standard Class": 0, "Second Class": 1, "First Class": 2, "Same Day": 3}
    shipping_mode_enc = float(mode_map.get(sample["shipping_mode"], 0))
    dept = sample["department"]
    
    supplier_delay_rate = float(df[df["department"] == dept]["late_risk"].mean())
    actual_days = float(sample["actual_days"])
    scheduled_days = float(sample["scheduled_days"])
    
    quantity = float(sample.get("quantity", 1))
    order_value = float(sample["order_value"])
    risk_features = {
        "shipping_mode_enc": shipping_mode_enc,
        "actual_days": actual_days,
        "scheduled_days": scheduled_days,
        "discount_rate": float(sample["discount_rate"]),
        "order_value": order_value,
        "supplier_delay_rate": supplier_delay_rate,
        "days_buffer": float(scheduled_days - max(0, actual_days)),
        "delay_gap": float(actual_days - scheduled_days)
    }
    
    # Get supplier metrics
    if supplier_df_cache is not None:
        supp_stats = supplier_df_cache[supplier_df_cache["department"] == dept]
        if not supp_stats.empty:
            supp_stats = supp_stats.iloc[0]
            supplier_metrics = {
                "avg_delivery_time": float(supp_stats["avg_delivery_time"]),
                "price_deviation": float(supp_stats["price_deviation"]),
                "fulfillment_rate": float(supp_stats["fulfillment_rate"]),
                "complaint_freq": float(supp_stats["complaint_freq"]),
            }
        else:
            supplier_metrics = SupplierMetricsInline().dict()
    else:
        supplier_metrics = SupplierMetricsInline().dict()
        
    # Pick random nodes for start and goal
    nodes = list(NODES.keys())
    start_node = int(np.random.choice(nodes))
    goal_node = int(np.random.choice([n for n in nodes if n != start_node]))
    
    inventory = float(np.random.randint(20, 200))
    days_to_delivery = scheduled_days

    if not state["models_loaded"]:
        raise HTTPException(status_code=503, detail="Models not yet ready. Wait for training to complete.")
    if daily_df_cache is None:
        raise HTTPException(status_code=503, detail="Feature data not loaded.")

    current_state = {
        "inventory": inventory,
        "days_to_delivery": days_to_delivery,
        "start_node": start_node,
        "goal_node": goal_node,
        "risk_features": risk_features,
        "supplier_metrics": supplier_metrics,
        "order_quantity": quantity,
        "avg_order_value_per_unit": float(df["order_value"].mean() / max(df["quantity"].mean(), 1)),
    }

    try:
        result = run_pipeline(
            xgb_model=xgb_model,
            rf_model=rf_model,
            iso_model=iso_model,
            q_table=q_table,
            daily_df=daily_df_cache,
            current_state=current_state,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {e}")

    city = str(sample.get("city", sample.get("order_city", "Unknown City")))
    country = str(sample.get("country", ""))
    order_id = str(sample.get("Order Id", sample.get("order_id", f"SIM-{np.random.randint(100000, 999999)}")))

    result["order_details"] = {
        "id": order_id,
        "city": city,
        "country": country,
        "value": order_value,
        "quantity": quantity,
        "shipping_mode": str(sample.get("shipping_mode", "Standard Class")),
        "department": str(dept),
        "inventory": inventory,
        "actual_days": actual_days,
        "scheduled_days": scheduled_days,
        "start_node": start_node,
        "goal_node": goal_node,
        "start_name": NODES[start_node][0],
        "goal_name": NODES[goal_node][0],
        "late_delivery_risk": int(sample.get("late_risk", 0)),
    }
    result["model_inputs"] = {
        "risk_features": risk_features,
        "supplier_metrics": supplier_metrics,
    }
    result["supplier_context"] = {
        "department": str(dept),
        "metrics": supplier_metrics,
        "is_anomaly": bool(result["anomaly"]["is_anomaly"]),
        "score": result["anomaly"]["score"],
        "delay_rate": round(supplier_delay_rate, 4),
    }
    result["history_row"] = {
        "order_id": order_id,
        "city": city,
        "department": str(dept),
        "value": round(order_value, 2),
        "risk": result["risk_level"],
        "anomaly": bool(result["anomaly"]["is_anomaly"]),
        "action": result["rl_action"]["action"],
        "route_type": result["route"].get("type", result["route"].get("algorithm", "")),
        "path": " -> ".join(result["route"].get("path_names", [])),
    }
    
    return result



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
