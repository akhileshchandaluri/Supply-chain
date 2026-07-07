import numpy as np
import random
import os

# ─── State-space bins ─────────────────────────────────────────────────────────
INV_BINS    = [0, 100, 300, 600, 1000, 2000]
DEMAND_BINS = [0, 100, 300, 600, 1000, 2000]
RISK_BINS   = [0.0, 0.3, 0.6, 0.9]
DAY_BINS    = [0, 1, 3, 7, 30]

# ─── Action space ─────────────────────────────────────────────────────────────
ACTIONS = {
    0: "HOLD",
    1: "REORDER_SMALL",
    2: "REORDER_MEDIUM",
    3: "REORDER_LARGE",
    4: "EMERGENCY_REORDER",
    5: "SWITCH_SUPPLIER",
}

# ─── Training hyperparameters (single source of truth) ────────────────────────
# Referenced by QLearningAgent defaults AND surfaced to the UI so the displayed
# values are the real ones the agent was trained with — not decorative numbers.
GAMMA         = 0.95   # discount factor in the Bellman update
ALPHA         = 0.1    # learning rate
EPSILON_START = 0.3    # initial exploration rate (ε-greedy)
EPSILON_MIN   = 0.01   # exploration floor after decay
EPSILON_DECAY = 0.995  # per-episode decay


def discretize(inv, demand, risk, anomaly, days) -> tuple:
    inv_b  = min(np.digitize(inv,    INV_BINS)    - 1, len(INV_BINS)    - 1)
    dem_b  = min(np.digitize(demand, DEMAND_BINS) - 1, len(DEMAND_BINS) - 1)
    risk_b = min(np.digitize(risk,   RISK_BINS)   - 1, len(RISK_BINS)   - 1)
    anom_b = int(bool(anomaly))
    days_b = min(np.digitize(days,   DAY_BINS)    - 1, len(DAY_BINS)    - 1)
    return (inv_b, dem_b, risk_b, anom_b, days_b)


class QLearningAgent:
    def __init__(self, alpha: float = ALPHA, gamma: float = GAMMA, epsilon: float = EPSILON_START):
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.eps_min   = EPSILON_MIN
        self.eps_decay = EPSILON_DECAY

        state_dims = (
            len(INV_BINS),
            len(DEMAND_BINS),
            len(RISK_BINS),
            2,
            len(DAY_BINS),
        )
        self.q_table = np.zeros((*state_dims, len(ACTIONS)))

    def choose_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next = float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


def simulate_environment(inv, demand, action_id, risk, anomaly):
    """One-step simulation of the supply chain environment."""
    reorder_qty = {
        0: 0,    # HOLD
        1: 150,  # REORDER_SMALL
        2: 400,  # REORDER_MEDIUM
        3: 800,  # REORDER_LARGE
        4: 600,  # EMERGENCY_REORDER
        5: 0,    # SWITCH_SUPPLIER
    }

    inv += reorder_qty.get(action_id, 0)
    inv -= demand  # consume demand

    reward = 0.0

    # Holding cost (penalty for excess inventory)
    reward -= inv * 0.05

    if inv < 0:
        reward -= 50.0  # Severe stockout penalty
        inv = 0
    else:
        reward += demand * 0.5  # Reward for fulfilling demand

    if inv > 1000:
        reward -= 20.0  # Overcapacity penalty

    if action_id == 4:  # EMERGENCY_REORDER
        reward -= 15.0
    elif action_id in [1, 2, 3]:  # Standard reorder cost
        reward -= 2.0

    if action_id == 5 and anomaly:  # SWITCH_SUPPLIER when anomaly present
        reward += 10.0

    return inv, reward


def explain_reward(inv, demand, action_id, risk=0.0, anomaly=0) -> dict:
    """
    Itemize the one-step reward for a state/action using the SAME rules as
    simulate_environment. Returns the reward components (so the UI can show a
    "points/reward breakdown") plus the resulting inventory and total.

    This is a transparent re-derivation of the reward function — not the trained
    Q-value (which is the discounted long-term expectation). Both are shown in
    the UI: reward_breakdown = why this step scores as it does; q_value = why the
    agent prefers this action over the long run.
    """
    reorder_qty = {0: 0, 1: 150, 2: 400, 3: 800, 4: 600, 5: 0}
    inv_after = inv + reorder_qty.get(action_id, 0) - demand

    components = {}
    # Holding cost on remaining inventory (applied on pre-clamp level, as in sim).
    components["holding_cost"] = round(-inv_after * 0.05, 3)

    if inv_after < 0:
        components["stockout_penalty"] = -50.0
        inv_after = 0
    else:
        components["demand_fulfillment"] = round(demand * 0.5, 3)

    if inv_after > 1000:
        components["overcapacity_penalty"] = -20.0

    if action_id == 4:  # EMERGENCY_REORDER
        components["action_cost"] = -15.0
    elif action_id in (1, 2, 3):  # standard reorder
        components["action_cost"] = -2.0

    if action_id == 5 and anomaly:  # SWITCH_SUPPLIER during an anomaly
        components["anomaly_mitigation"] = 10.0

    total = round(float(sum(components.values())), 3)
    return {
        "components": components,
        "total": total,
        "projected_inventory": round(float(inv_after), 2),
    }


def train_agent(episodes: int = 10000, save_path: str = "models/q_table.npy"):
    agent = QLearningAgent()
    episode_rewards = []

    for ep in range(episodes):
        inv     = random.randint(100, 800)
        demand  = random.randint(50, 600)
        risk    = random.uniform(0, 1)
        anomaly = random.choice([0, 1])
        days    = random.randint(1, 14)

        state = discretize(inv, demand, risk, anomaly, days)
        total_reward = 0.0

        for _ in range(30):
            action = agent.choose_action(state)
            inv, reward = simulate_environment(inv, demand, action, risk, anomaly)

            demand  = max(0, demand + random.randint(-50, 50))
            risk    = min(1.0, max(0.0, risk + random.uniform(-0.1, 0.1)))
            if action == 5:
                anomaly = 0
            else:
                anomaly = random.choice([0, 1]) if random.random() < 0.1 else anomaly
            days = max(1, days - 1)

            # Simulate a supplier delivery resetting the days if we did a reorder
            if action in [1, 2, 3, 4]:
                days = random.randint(3, 7)

            next_state = discretize(inv, demand, risk, anomaly, days)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % 1000 == 0:
            avg = float(np.mean(episode_rewards[-1000:]))
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Avg Reward: {avg:.2f} | ε: {agent.epsilon:.3f}"
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, agent.q_table)
    print(f"\nTraining complete. Q-table saved to {save_path}")
    return agent, episode_rewards


def get_action(q_table: np.ndarray, inv, demand, risk, anomaly, days) -> dict:
    """
    Inference — return the greedy best action for the current state, plus the
    real decision internals used by the RL Insights UI.

    All fields are derived from the trained Q-table (no fabricated numbers):
      - q_value    : Q(s,a) of the chosen action, straight from the table
      - confidence : softmax probability of the chosen action over this state's
                     Q-values — a genuine measure of how decisively the agent
                     prefers this action vs. the alternatives
      - hyperparameters: the actual training config (gamma/epsilon), so the UI
                     can display the Bellman discount and exploration rate truthfully
    """
    state = discretize(inv, demand, risk, anomaly, days)
    q_row = np.asarray(q_table[state], dtype=float)
    action_id = int(np.argmax(q_row))
    q_values = q_row.tolist()

    # Softmax over the state's Q-values → decision confidence for the chosen action.
    shifted = q_row - np.max(q_row)
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    confidence = float(exp[action_id] / denom) if denom > 0 else 1.0 / len(q_values)

    return {
        "action": ACTIONS[action_id],
        "action_id": action_id,
        "q_value": round(float(q_row[action_id]), 3),
        "confidence": round(confidence, 4),
        "q_values": {ACTIONS[i]: round(q_values[i], 3) for i in range(len(ACTIONS))},
        # Itemized one-step reward for the chosen action (additive field).
        "reward_breakdown": explain_reward(inv, demand, action_id, risk, anomaly),
        "hyperparameters": {
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_min": EPSILON_MIN,
            "policy": "greedy (argmax) at inference; ε-greedy during training",
        },
    }
