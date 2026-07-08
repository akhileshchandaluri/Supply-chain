# SmartChain AI - Comprehensive Architecture & Coding Documentation

This document serves as the complete technical blueprint for the **SmartChain AI** supply chain optimization project. It covers the high-level architecture, the 4-stage AI pipeline, the backend structure, the frontend interface, and the integration orchestrator.

---

## 1. High-Level Architecture Overview

SmartChain AI is built as a highly modular, decoupled web application that bridges the gap between machine learning and operational logistics:
- **Frontend**: A React single-page application (SPA) built with Vite. It features dynamic charts, interactive geographic logistics maps, and real-time dashboards to visualize AI decisions.
- **Backend**: A high-performance Python FastAPI server acting as the central intelligence hub, capable of handling concurrent simulation requests.
- **Intelligence Layer**: A sequence of 4 specialized machine learning (ML) and operations research (OR) models chained together to simulate, forecast, and optimize supply chain operations.
- **Explainable AI (XAI)**: An LLM-powered layer (Groq API) that translates complex mathematical outputs, Q-table rewards, and LP allocations from the models into human-readable executive audits.

---

## 2. The 4-Stage AI Master Pipeline

The core orchestration logic resides in `backend/src/integration.py`. The `run_pipeline` function sequentially passes environment state and outputs across four disparate systems. Each stage feeds into the next, culminating in a single unified decision block.

### Stage 1: Demand Forecasting (XGBoost)
- **File**: `backend/src/xgboost_demand.py`
- **Role**: Predicts inventory demand for the next 7 calendar days to drive procurement and warehouse allocation.
- **Architecture**: A tuned XGBoost Regressor (`XGBRegressor`). 
  - **Hyperparameters**: Tuned for short-series generalization with `n_estimators=600`, `learning_rate=0.03`, `max_depth=4`, and `L2 regularization (reg_lambda=2.0)`.
  - **Feature Engineering**: Operates on 12 engineered time-series features including `rolling_7d_mean`, `lag_1`, `lag_7`, `lag_30`, and temporal indicators (`day_of_week`, `quarter`).
  - **Residual Delta Approach**: It prevents tree-based extrapolation failure by predicting the *residual delta* (the change from the last known demand `lag_1`) rather than absolute future demand. The absolute demand is reconstructed post-prediction.
- **Output**: An array of 7 floating-point values representing future demand quantities.

### Stage 2: Risk Classification (Random Forest)
- **File**: `backend/src/rf_risk.py`
- **Role**: Evaluates the probability of late delivery and categorizes risk into `LOW`, `MEDIUM`, or `HIGH`.
- **Architecture**: A Random Forest Classifier (`RandomForestClassifier(n_estimators=100)`).
  - **Feature Inputs**: Evaluates 8 core features on a per-order basis: `shipping_mode_enc`, `actual_days`, `scheduled_days`, `discount_rate`, `order_value`, `supplier_delay_rate`, `days_buffer`, and `delay_gap`.
  - **Evaluation**: The model outputs a probability array which maps to the discrete risk classes.
- **Output**: A discrete risk label (`LOW/MEDIUM/HIGH`) and a normalized numeric risk score (0.2 for LOW, 0.55 for MEDIUM, 0.85 for HIGH) which acts as an environment variable for the RL agent.

### Stage 3: Inventory Action Decision (Q-Learning RL Agent)
- **File**: `backend/src/rl_agent.py`
- **Role**: The ultimate decision-maker for inventory management, determining how the supply chain should react to the current forecast and risk profile.
- **Architecture**: Reinforcement Learning using a Q-Learning algorithm. 
  - **State Space**: It maintains a 6-dimensional state space tensor `(6, 6, 4, 2, 5, 6)` derived from discretizing: Inventory levels, XGBoost Demand Forecast, RF Risk Score, Delivery Days, and external penalties.
  - **Reward Function**: Calculates operational ROI by heavily penalizing stockouts (lost sales) and emergency air-freight costs, while gently penalizing excess warehouse carrying costs.
- **Output**: One of four optimal actions based on the highest Q-value: 
  - `DO_NOTHING`
  - `STANDARD_REORDER`
  - `EMERGENCY_REORDER`
  - `LIQUIDATE`

### Stage 3.5: Physical Warehouse Allocation (GLOP LP)
- **File**: `backend/src/optimization_layer.py`
- **Role**: Bridges the macro-decision from the RL agent (e.g., `STANDARD_REORDER`) to a micro-level physical warehouse reality.
- **Architecture**: Uses Google OR-Tools (GLOP Linear Programming Solver). 
  - **Constraints**: Formulates a linear programming problem constrained by warehouse capacities, regional demand requirements, and fixed transport costs.
  - **Objective**: Minimizes the combined cost function of storage ($/unit) and transit ($/km).
- **Output**: An optimal unit allocation mapping (e.g., Warehouse A ships 200 units, Warehouse B ships 50 units).

### Stage 4: Routing & Logistics (A* Algorithm)
- **File**: `backend/src/astar_routing.py`
- **Role**: Calculates the exact path for the shipment through the logistics network.
- **Architecture**: Operates on a defined graph of 18 nodes (Warehouses, Hubs, Ports, Cities).
  - **Algorithm**: Implements the `A* (A-Star)` pathfinding algorithm.
  - **Heuristic**: Uses geographic coordinate distances (Haversine/Euclidean approximations) as the heuristic to guide the search towards the destination node efficiently.
- **Heuristic Logic**: If the RL agent triggers an `EMERGENCY_REORDER`, or if the Random Forest Risk Score is > 0.8, it logs the route as an **EMERGENCY (A*)** priority route to alert the frontend mapping component to highlight the path in red.

---

## 3. Backend Codebase & Directory Structure

The backend operates on a strict modular separation of concerns.

```text
backend/
├── data/
│   └── DataCoSupplyChainDataset.csv  # The raw dataset driving all ML training
├── models/                           # Auto-generated saved models (.pkl / .npy)
├── src/
│   ├── data_preprocessing.py         # Cleans raw CSV (dates, renames, drops nulls)
│   ├── feature_engineering.py        # Transforms raw data into ML-ready time-series features
│   ├── xgboost_demand.py             # Model 1: Demand Prediction
│   ├── rf_risk.py                    # Model 2: Risk Classification
│   ├── rl_agent.py                   # Model 3: Q-Learning Agent
│   ├── optimization_layer.py         # Model 3.5: Google OR-Tools GLOP
│   ├── astar_routing.py              # Model 4: A* Pathfinding
│   ├── graph_construction.py         # Logistics network definition (18 hardcoded nodes)
│   ├── integration.py                # Master orchestrator chaining Models 1-4
│   └── xai_explainer.py              # Groq LLM integration for executive audits
├── main.py                           # FastAPI Server Entrypoint & API Routes
└── evaluate_all.py                   # Metrics generation script
```

### Key API Endpoints (`main.py`)
- `POST /api/pipeline/run`: Triggers the master orchestrator manually with a JSON payload of defined states (Inventory, Order Value, Nodes).
- `GET /api/simulate/order`: The primary demo endpoint. It pulls a random historic row from the dataset, passes it through the active models, injects realistic noise (via RNG seeding), and returns the full pipeline output.
- `POST /api/rl/chat`: Facilitates the XAI chatbot by feeding the latest pipeline context JSON to the Groq LLM.
- `POST /api/train`: Background task endpoint to re-compile and retrain all models asynchronously, ensuring server availability during the heavy compute phase.

---

## 4. Frontend Codebase & Interface

The frontend is a `React + Vite` application styled with raw vanilla CSS (`index.css`) emphasizing deep glassmorphism and modern dashboard aesthetics.

```text
frontend/
├── src/
│   ├── api/
│   │   └── client.js                 # Axios wrapper for FastAPI communication
│   ├── components/
│   │   ├── AllocationResultTable.jsx # Renders GLOP LP results with cost breakdowns
│   │   ├── RLChatPanel.jsx           # Renders the Groq XAI chatbot interface
│   │   ├── RLInsightsPanel.jsx       # Renders Q-Learning reward breakdowns
│   │   └── Sidebar.jsx               # Navigation structure
│   ├── pages/
│   │   ├── Dashboard.jsx             # Main Simulation Hub (Calls /simulate/order)
│   │   ├── DemandForecast.jsx        # XGBoost Deep-dive View with line charts
│   │   ├── RiskAnomaly.jsx           # Random Forest Deep-dive View
│   │   ├── RLAgent.jsx               # Q-Learning Deep-dive View with state mapping
│   │   └── RouteOptimization.jsx     # Network Graph View for A* Routing
│   ├── App.jsx                       # React Router configuration
│   └── index.css                     # Global Design System (Tokens, Animations)
```

### Aesthetic Philosophy
The UI rejects utility frameworks like Tailwind in favor of a curated CSS design system defined in `index.css`. It features:
- Deep dark mode `#0f172a` backgrounds with dynamic neon glows (e.g., `#3b82f6` for primary actions, `#ef4444` for emergency alerts).
- Micro-animations (fade-ins, subtle shifts on hover, glowing borders) to make the dashboard feel alive, interactive, and premium.
- Clean component lifecycle management to prevent chart flickering during data re-fetches.

---

## 5. Development & Deployment Workflow

1. **Auto-Training Protocol**: On startup, `main.py` utilizes the FastAPI `lifespan` context manager to check if all model binaries (`.pkl` and `.npy`) exist in `backend/models/`. If any are missing, it spawns a background thread to trigger an asynchronous training sequence using the dataset, allowing the server to boot immediately.
2. **Stateless Operations**: The models predict without maintaining internal state (besides the RL Q-Table matrices and XGBoost lag dependencies). This ensures FastAPI remains scalable, thread-safe, and capable of horizontal scaling.
3. **Execution**: Start the backend via `uvicorn main:app --reload` and the frontend via `npm run dev`.
