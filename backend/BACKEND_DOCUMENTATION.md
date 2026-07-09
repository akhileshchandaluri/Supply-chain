# AI Supply Chain Backend - Master Documentation

This is the ultimate, exhaustive guide to the entire Python backend of the AI Supply Chain project. It includes both high-level explanations of what each file does, alongside actual code snippets and detailed breakdowns of the underlying math and logic. Use this to master the codebase and confidently answer any technical questions.

---

## 1. `backend/main.py` (The API Server)

**High-Level Purpose:**
This is the core FastAPI web server. It listens for requests from the React frontend, coordinates the execution of the AI models, and sends the final JSON response back. It also handles the loading of the CSV dataset and the trained AI models into memory on startup.

### Key Code & Explanations

**Startup Event & Data Loading:**
```python
@app.on_event("startup")
def startup_event():
    global raw_df_cache, supplier_df_cache, daily_df_cache
    global xgb_model, rf_model, iso_model, q_table
    # ...
    raw_df_cache = load_data(DATA_PATH)
    supplier_df_cache = build_supplier_features(raw_df_cache)
    daily_df_cache = aggregate_daily_demand(raw_df_cache)
    
    load_models()
```
**Explanation:** When you run `python main.py`, this function triggers. It reads the massive `DataCoSupplyChainDataset.csv`, preprocesses it into usable features (`daily_df_cache`), and loads all the trained AI models (`.pkl` and `.npy` files) into RAM. This ensures that when the user clicks "Execute", the API responds instantly without having to read from the hard drive.

**Custom Simulation Endpoint:**
```python
@app.post("/api/pipeline/run")
def run_custom_pipeline(req: PipelineRequest):
    # Build current_state dict that integration.py expects
    current_state = {
        "inventory":        req.inventory,
        "days_to_delivery": req.days_to_delivery,
        "start_node":       req.start_node,
        "goal_node":        req.goal_node,
        "order_quantity":   req.order_quantity,
        # ...
    }

    result = run_pipeline(xgb_model, rf_model, iso_model, q_table, daily_df_cache, current_state)
    
    # Inject order details so the frontend has inventory context
    result["order_details"] = {
        "inventory": float(req.inventory),
        "quantity": float(req.order_quantity),
    }

    return result
```
**Explanation:** This is what runs when you use the "Custom Live Order" form in the UI. It pulls the numbers you typed in (like `req.inventory = 120`), packages them into a `current_state` dictionary, and hands it off to `integration.py` which runs all 5 AI models. It then injects the `order_details` back into the final result so the React UI can display your custom numbers correctly.

---

## 2. `backend/src/integration.py` (The Master AI Pipeline)

**High-Level Purpose:**
This file acts as the "Conductor" of the orchestra. It takes the raw order data and passes it sequentially through all 5 AI models (XGBoost -> Random Forest -> Isolation Forest -> RL Agent -> OR-Tools/A*).

### Key Code & Explanations

**The 5-Stage Execution Sequence:**
```python
def run_pipeline(xgb_model, rf_model, iso_model, q_table, daily_df, current_state):
    # ── Step 1: XGBoost (Demand Forecast) ──
    forecast = predict_demand(xgb_model, daily_df, horizon=7)
    demand_7d = float(np.mean(forecast))

    # ── Step 2: Random Forest (Risk Assessment) ──
    risk_result = predict_risk(rf_model, risk_features)
    risk_label  = risk_result["label"]
    risk_score  = _RISK_SCORE_MAP.get(risk_label, 0.5)

    # ── Step 3: Isolation Forest (Supplier Anomaly) ──
    anomaly_result = detect_anomaly(iso_model, supplier_metrics)
    anomaly_flag   = int(anomaly_result["is_anomaly"])

    # ── Step 4: Q-Learning RL (Inventory Decision) ──
    rl_result = get_action(q_table, inv=current_state.get("inventory"), demand=demand_7d, risk=risk_score, anomaly=anomaly_flag, days=...)
```
**Explanation:** Notice how the outputs of the earlier models feed directly into the Reinforcement Learning agent! 
1. XGBoost predicts the `demand_7d`.
2. Random Forest predicts the `risk_score`.
3. Isolation Forest outputs the `anomaly_flag`.
4. All of these variables are directly plugged into `get_action(...)` so the Q-Learning RL Agent can make a mathematically informed decision using the intelligence gathered by the other 3 models.

---

## 3. `backend/src/rl_agent.py` (The Reinforcement Learning Brain)

**High-Level Purpose:**
This is the most complex AI in the project. It uses Q-Learning (a type of Reinforcement Learning) where an AI agent learns to manage inventory through millions of trial-and-error simulations. 

### Key Code & Explanations

**State Discretization (Bucketing):**
```python
INV_BINS    = [0, 100, 300, 600, 1000, 2000]
DEMAND_BINS = [0, 100, 300, 600, 1000, 2000]

def discretize(inv, demand, risk, anomaly, days) -> tuple:
    inv_b  = min(np.digitize(inv,    INV_BINS)    - 1, len(INV_BINS)    - 1)
    # ...
    return (inv_b, dem_b, risk_b, anom_b, days_b)
```
**Explanation:** A Q-Table is a literal grid (matrix). It cannot handle infinite decimals (like an inventory of 143.52). The `np.digitize` function places the raw inventory number into a "bucket". If inventory is 150, it falls into the `[100-300]` bucket, becoming State Level 2. This compresses millions of numbers into a finite, trainable grid.

**The Reward Function (The Math of the Environment):**
```python
def simulate_environment(inv, demand, action_id, risk, anomaly):
    # e.g., Action 2 (REORDER_MEDIUM) adds 400 to inventory
    inv += reorder_qty.get(action_id, 0)
    inv -= demand  

    reward = 0.0
    reward -= inv * 0.05  # Holding cost penalty

    if inv < 0:
        reward -= 50.0    # Massive Stockout penalty
    else:
        reward += demand * 0.5  # Demand fulfillment reward
```
**Explanation:** This defines the "game" the AI is playing. When the AI takes an action (ordering stock), we simulate the inventory going up, and then immediately subtract the `demand`. 
* If inventory drops below 0, it means we failed a customer, so we heavily penalize the AI (`-50.0`). 
* If the AI successfully covers the demand, it gets a reward (`+0.5` per unit).
* The AI learns over 10,000 episodes to maximize this `reward` variable.

**The Bellman Equation (Q-Table Update):**
```python
    def update(self, state, action, reward, next_state):
        best_next = float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
```
**Explanation:** This is the famous Bellman Equation. When the AI takes an action and receives a `reward`, it updates its brain (`self.q_table`). 
* `best_next` calculates the maximum possible reward it can get in the *next* state.
* `td_error` calculates the difference between what the AI *thought* the reward would be, and what it *actually* was.
* `self.alpha` is the learning rate, which overwrites old beliefs with new knowledge.

---

## 4. `backend/src/xai_explainer.py` (Explainable AI / Chat)

**High-Level Purpose:**
Uses a Large Language Model (Llama 3.3 via Groq) to bridge the gap between complex AI math and human operators, providing plain-English executive summaries and a live chat interface.

### Key Code & Explanations

**Interactive Chat Logic:**
```python
def chat_about_decision(question: str, context: dict, history=None) -> str:
    system_instruction = f"""
    You are the AI Supply Chain Analyst. Answer the user's question.
    Context of the current RL decision: {json.dumps(context)}
    """
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question}
        ]
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]
```
**Explanation:** When you use the "AI Core" chat in the UI, your question hits this function. Crucially, we use `json.dumps(context)` to embed the massive JSON payload (containing current inventory, the math formula for projected inventory, Q-Values, and demand forecasts) directly into the hidden `system` prompt. Because the Llama 3.3 model reads this hidden context *before* answering, it has perfect knowledge of the mathematical state of the supply chain and doesn't have to guess!

---

## 5. `backend/src/optimization_layer.py` (OR-Tools / Google GLOP)

**High-Level Purpose:**
While the RL Agent decides *how much* to order, this script mathematically calculates *where* to order it from to save money.

### Key Code & Explanations

**Linear Programming (Min-Cost Flow):**
```python
def optimize_allocation(target_quantity, risk_level):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    
    # Variables: how much to order from each warehouse
    x = {}
    for w in warehouses:
        x[w['id']] = solver.NumVar(0, w['capacity'], w['name'])

    # Constraint: The sum of allocations must exactly equal the target quantity
    solver.Add(sum(x[w['id']] for w in warehouses) == target_quantity)

    # Objective: Minimize total cost
    objective = solver.Objective()
    for w in warehouses:
        cost_multiplier = 1.5 if risk_level == "HIGH" else 1.0
        total_unit_cost = w['cost_per_unit'] + (w['shipping_cost'] * cost_multiplier)
        objective.SetCoefficient(x[w['id']], total_unit_cost)
    
    objective.SetMinimization()
    status = solver.Solve()
```
**Explanation:** 
* `pywraplp.Solver.CreateSolver('GLOP')` creates a Google Linear Programming solver.
* `solver.NumVar(0, w['capacity'])` prevents the system from ordering more stock than a warehouse actually has.
* `solver.Add(...)` forces the equation to exactly equal the `target_quantity` requested by the RL Agent.
* `objective.SetCoefficient` sets the cost per unit. Notice that if the Random Forest flagged the route as `HIGH` risk, the shipping cost mathematically spikes by 1.5x (`cost_multiplier`), forcing the algorithm to buy from a safer warehouse if it's cheaper overall.
* `objective.SetMinimization()` mathematically proves and selects the absolute cheapest route.

---

## 6. `backend/src/rf_risk.py` & `isolation_forest.py` & `xgboost_demand.py`

**High-Level Purpose:**
These files are standard Machine Learning predictive models built using `scikit-learn` and `xgboost`. 

### Key Code & Explanations

**Random Forest (rf_risk.py):**
```python
def train_rf_model(df):
    features = ['shipping_mode_enc', 'actual_days', 'scheduled_days', 'discount_rate', 'order_value']
    # Target: 1 if actual_days > scheduled_days (Late), else 0
    y = (df['actual_days'] > df['scheduled_days']).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(df[features], y)
```
**Explanation:** We train 100 decision trees (`n_estimators=100`) to predict if a delivery will be late. It looks at shipping mode, actual days vs scheduled days, and order values. At runtime, it outputs a probability (e.g. 85% chance of being late).

**Isolation Forest (isolation_forest.py):**
```python
def train_isolation_forest(df):
    features = ['avg_delivery_time', 'price_deviation', 'fulfillment_rate', 'complaint_freq']
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[features])
```
**Explanation:** Unsupervised learning. It randomly slices data. Normal suppliers cluster together; anomalies are isolated quickly. If a supplier starts failing, they are flagged as an anomaly. `contamination=0.05` means we assume 5% of our historical supplier data is fraudulent/anomalous.

**XGBoost (xgboost_demand.py):**
```python
def train_xgboost(daily_df):
    # Uses lag features (demand from 1 day ago, 7 days ago) to predict future demand
    features = ['lag_1', 'lag_7', 'rolling_mean_3', 'day_of_week']
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
```
**Explanation:** Uses Gradient Boosting. It relies heavily on "lag features" (e.g., what was the demand exactly 7 days ago?). Machine learning models need lag features to understand time-series trends and seasonality.

---

## 7. `backend/src/astar_routing.py` & `graph_construction.py` (The Routing Engine)

**High-Level Purpose:**
These files map out the physical geographical path the truck/ship should take using the A* (A-Star) Pathfinding Algorithm.

### Key Code & Explanations

**A-Star Algorithm:**
```python
def astar(start_node, goal_node):
    # f(n) = g(n) + h(n)
    # g_score: actual distance from start to current node
    # f_score: g_score + heuristic (straight-line distance to goal)
    
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal_node:
            return reconstruct_path(came_from, current)
            
        for neighbor, weight in get_neighbors(current):
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
```
**Explanation:** 
* `graph_construction.py` builds the map (nodes and edges with Lat/Lon coordinates).
* A* evaluates paths using `f(n) = g(n) + h(n)`.
* `g(n)` is the actual distance traveled so far (`tentative_g_score`).
* `h(n)` is the "heuristic" (the straight line distance to the destination).
* By constantly pushing the lowest `f_score` to a priority queue (`heapq`), it guarantees finding the shortest physical route between the warehouse and the destination hub without blindly searching every single city.
