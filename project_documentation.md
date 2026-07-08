# SmartChain AI - Comprehensive Architecture & Coding Documentation

This document serves as the complete technical blueprint for the **SmartChain AI** supply chain optimization project. It covers the high-level architecture, the 5-stage AI pipeline, the backend structure, the frontend interface, and the integration orchestrator. 

---

## 1. High-Level Architecture Overview

SmartChain AI is built as a modern, decoupled web application:
- **Frontend**: A React single-page application (SPA) built with Vite, featuring dynamic charts, interactive maps, and real-time dashboards.
- **Backend**: A high-performance Python FastAPI server acting as the central intelligence hub.
- **Intelligence Layer**: A sequence of 5 specialized machine learning and operations research models chained together to simulate, forecast, and optimize supply chain operations.
- **Explainable AI (XAI)**: An LLM-powered layer (Groq API) that translates complex mathematical outputs from the models into human-readable executive audits.

---

## 2. The 5-Stage AI Master Pipeline

The core logic resides in `backend/src/integration.py`. This orchestrator (`run_pipeline`) sequentially passes state and output across five disparate systems:

### Stage 1: Demand Forecasting (XGBoost)
- **File**: `backend/src/xgboost_demand.py`
- **Role**: Predicts inventory demand for the next 7 calendar days.
- **Architecture**: A tuned XGBoost Regressor (`XGBRegressor`). It prevents extrapolation failure by predicting the *residual delta* (change from the last known demand `lag_1`) rather than absolute future demand. It uses a 7-day rolling evaluation metric.
- **Output**: An array of 7 values representing future demand quantities.

### Stage 2: Risk Classification (Random Forest)
- **File**: `backend/src/rf_risk.py`
- **Role**: Evaluates the probability of late delivery and categorizes risk into `LOW`, `MEDIUM`, or `HIGH`.
- **Architecture**: A Random Forest Classifier trained on 8 core features (e.g., shipping mode, actual vs. scheduled days, discount rate, supplier delay rate).
- **Output**: Risk label and a normalized numeric risk score (0.2 for LOW, 0.55 for MEDIUM, 0.85 for HIGH) to be consumed by the RL agent.

### Stage 3: Supplier Anomaly Detection (Isolation Forest)
- **File**: `backend/src/isolation_forest.py`
- **Role**: Flags suppliers exhibiting anomalous behaviors (e.g., extreme price deviation, frequent complaints, poor fulfillment).
- **Architecture**: An Isolation Forest (unsupervised anomaly detection). It maps metrics into a binary anomaly flag.
- **Output**: Boolean `is_anomaly` flag and a continuous anomaly score.

### Stage 4: Inventory Action Decision (Q-Learning RL Agent)
- **File**: `backend/src/rl_agent.py`
- **Role**: The ultimate decision-maker for inventory management.
- **Architecture**: Reinforcement Learning (Q-Learning). It maintains a 6-dimensional state space tensor `(6, 6, 4, 2, 5, 6)` mapping Inventory, Demand, Risk, Anomaly flags, and Delivery Days to optimal actions.
- **Output**: One of four actions: `DO_NOTHING`, `STANDARD_REORDER`, `EMERGENCY_REORDER`, `LIQUIDATE`.

### Stage 4.5: Physical Warehouse Allocation (GLOP LP)
- **File**: `backend/src/optimization_layer.py`
- **Role**: Bridges the macro-decision from the RL agent to a micro-level physical reality.
- **Architecture**: Uses Google OR-Tools (GLOP Linear Solver). Given a reorder decision, it optimally allocates the requested inventory across available warehouses to minimize total carrying and transport costs.

### Stage 5: Routing & Logistics (A* / Dijkstra)
- **File**: `backend/src/astar_routing.py` and `backend/src/dijkstra_routing.py`
- **Role**: Calculates the path for the shipment through the logistics network.
- **Architecture**: Operates on a defined graph of 18 nodes (Warehouses, Hubs, Ports, Cities).
- **Heuristic Logic**: If the RL agent triggers an `EMERGENCY_REORDER`, or if Risk is >0.8, or if an anomaly is detected, it forces an `A*` emergency route. Otherwise, it defaults to standard shortest-path logic.

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
│   ├── xgboost_demand.py             # Model 1 logic
│   ├── rf_risk.py                    # Model 2 logic
│   ├── isolation_forest.py           # Model 3 logic
│   ├── rl_agent.py                   # Model 4 logic
│   ├── optimization_layer.py         # Model 4.5 logic (OR-Tools)
│   ├── astar_routing.py              # Model 5 logic
│   ├── dijkstra_routing.py           # Model 5 logic
│   ├── graph_construction.py         # Logistics network definition (18 hardcoded nodes)
│   ├── integration.py                # Master orchestrator chaining Models 1-5
│   └── xai_explainer.py              # Groq LLM integration for executive audits
├── main.py                           # FastAPI Server Entrypoint & API Routes
└── evaluate_all.py                   # Metrics generation script
```

### Key API Endpoints (`main.py`)
- `POST /api/pipeline/run`: Triggers the master orchestrator manually with a defined state.
- `GET /api/simulate/order`: The primary demo endpoint. It pulls a random historic row from the dataset, passes it through the active models, injects realistic noise, and returns the full pipeline output.
- `POST /api/rl/chat`: Facilitates the AI chatbot by feeding the latest pipeline context to the XAI Explainer.
- `POST /api/train`: Background task endpoint to re-compile and retrain all 4 models asynchronously.

---

## 4. Frontend Codebase & Interface

The frontend is a `React + Vite` application styled with raw vanilla CSS (`index.css`) emphasizing deep glassmorphism and modern dashboard aesthetics.

```text
frontend/
├── src/
│   ├── api/
│   │   └── client.js                 # Axios wrapper for FastAPI communication
│   ├── components/
│   │   ├── AllocationResultTable.jsx # Renders GLOP LP results
│   │   ├── RLChatPanel.jsx           # Renders the Groq XAI chatbot interface
│   │   ├── RLInsightsPanel.jsx       # Renders Q-Learning reward breakdowns
│   │   └── Sidebar.jsx               # Navigation structure
│   ├── pages/
│   │   ├── Dashboard.jsx             # Main Simulation Hub (Calls /simulate/order)
│   │   ├── DemandForecast.jsx        # XGBoost Deep-dive View
│   │   ├── RiskAnomaly.jsx           # RF & Isolation Forest Deep-dive View
│   │   ├── RLAgent.jsx               # Q-Learning Deep-dive View
│   │   └── RouteOptimization.jsx     # Network Graph View (Leaflet / D3)
│   ├── App.jsx                       # React Router configuration
│   └── index.css                     # Global Design System (Tokens, Animations)
```

### Aesthetic Philosophy
The UI rejects Tailwind in favor of a curated CSS design system in `index.css`. It features:
- Deep dark mode `#0f172a` backgrounds with dynamic neon glows (e.g., `#3b82f6` for primary actions).
- Micro-animations (fade-ins, subtle shifts on hover) to make the dashboard feel alive and interactive.

---

## 5. Development & Deployment Workflow

1. **Auto-Training**: On startup, `main.py` checks if all 4 model binaries (`.pkl` and `.npy`) exist in `backend/models/`. If any are missing, it triggers an asynchronous training sequence using the dataset.
2. **Stateless Operations**: The models predict without maintaining internal state (besides the RL Q-Table and XGBoost lag dependencies). This ensures FastAPI remains scalable and thread-safe.
3. **Execution**: Start the backend via `uvicorn main:app --reload` and the frontend via `npm run dev`.
