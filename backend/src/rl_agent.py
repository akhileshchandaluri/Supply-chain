import numpy as np
import random
import os

# ─── State-space bins ─────────────────────────────────────────────────────────
INV_BINS    = [0, 50, 150, 300, 500]
DEMAND_BINS = [0, 30, 80, 150, 300]
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


def discretize(inv, demand, risk, anomaly, days) -> tuple:
    inv_b  = min(np.digitize(inv,    INV_BINS)    - 1, len(INV_BINS)    - 1)
    dem_b  = min(np.digitize(demand, DEMAND_BINS) - 1, len(DEMAND_BINS) - 1)
    risk_b = min(np.digitize(risk,   RISK_BINS)   - 1, len(RISK_BINS)   - 1)
    anom_b = int(bool(anomaly))
    days_b = min(np.digitize(days,   DAY_BINS)    - 1, len(DAY_BINS)    - 1)
    return (inv_b, dem_b, risk_b, anom_b, days_b)


class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.3):
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.eps_min   = 0.01
        self.eps_decay = 0.995

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
        1: 50,   # REORDER_SMALL
        2: 150,  # REORDER_MEDIUM
        3: 300,  # REORDER_LARGE
        4: 200,  # EMERGENCY_REORDER
        5: 0,    # SWITCH_SUPPLIER
    }

    inv += reorder_qty.get(action_id, 0)
    inv -= demand  # consume demand

    reward = 0.0
    if 50 <= inv <= 300:
        reward += 10.0
    if inv < 0:
        reward -= 5.0
        inv = 0
    if inv > 300:
        reward -= 3.0
    if action_id == 4:  # EMERGENCY_REORDER
        reward -= 8.0
    if action_id == 5 and anomaly:  # SWITCH_SUPPLIER when anomaly present
        reward += 5.0

    return inv, reward


def train_agent(episodes: int = 10000, save_path: str = "models/q_table.npy"):
    agent = QLearningAgent()
    episode_rewards = []

    for ep in range(episodes):
        inv     = random.randint(50, 300)
        demand  = random.randint(20, 150)
        risk    = random.uniform(0, 1)
        anomaly = random.choice([0, 1])
        days    = random.randint(1, 14)

        state = discretize(inv, demand, risk, anomaly, days)
        total_reward = 0.0

        for _ in range(30):
            action = agent.choose_action(state)
            inv, reward = simulate_environment(inv, demand, action, risk, anomaly)

            demand  = max(0, demand + random.randint(-20, 20))
            risk    = min(1.0, max(0.0, risk + random.uniform(-0.1, 0.1)))
            anomaly = random.choice([0, 1]) if random.random() < 0.1 else anomaly
            days    = max(1, days - 1)

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
    """Inference — return best action for current state."""
    state = discretize(inv, demand, risk, anomaly, days)
    action_id = int(np.argmax(q_table[state]))
    q_values = q_table[state].tolist()
    return {
        "action": ACTIONS[action_id],
        "action_id": action_id,
        "q_values": {ACTIONS[i]: round(q_values[i], 3) for i in range(len(ACTIONS))},
    }
