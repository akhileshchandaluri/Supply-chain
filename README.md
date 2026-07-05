# SmartChain AI

**Intelligent Supply Chain Management System**  
RV College of Engineering | Department of AIML | Course: AI244AI

---

## Quick Start

### 1. Install Python Dependencies
```bash
cd smartchain_ai/backend
pip install -r requirements.txt
```

### 2. Add Dataset
Download `DataCoSupplyChainDataset.csv` from Kaggle:  
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis  
Place it at: `backend/data/DataCoSupplyChainDataset.csv`

### 3. Start Backend
```bash
cd smartchain_ai/backend
python main.py
# → API running at http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

### 4. Start Frontend (new terminal)
```bash
cd smartchain_ai/frontend
npm install
npm run dev
# → UI running at http://localhost:5173
```

### 5. Train Models
Open the dashboard at http://localhost:5173, click **"Train Models"**.  
Training runs in the background (~5–10 min). Poll the status indicator in the sidebar.

---

## Architecture

| Layer | Component | Algorithm |
|---|---|---|
| ML Layer 1 | `src/xgboost_demand.py` | XGBoost demand forecasting |
| ML Layer 2 | `src/rf_risk.py` | Random Forest risk classification (LOW/MED/HIGH) |
| ML Layer 3 | `src/isolation_forest.py` | Isolation Forest anomaly detection |
| AI Layer 1 | `src/astar_routing.py` | A* search (Haversine heuristic) |
| AI Layer 2 | `src/dijkstra_routing.py` | Dijkstra cost-optimal routing |
| AI Layer 3 | `src/rl_agent.py` | Q-Learning RL inventory agent |
| API | `main.py` | FastAPI REST server |
| UI | `frontend/` | React + Vite SPA |

## Expected Metrics
| Model | Metric | Target |
|---|---|---|
| XGBoost | RMSE | ~23.4 |
| XGBoost | R² | ~0.89 |
| Random Forest | Accuracy | ~87.3% |
| Random Forest | F1 | ~0.87 |
| Random Forest | ROC-AUC | ~0.91 |
| A* vs Dijkstra | Nodes explored | ~34% fewer |
