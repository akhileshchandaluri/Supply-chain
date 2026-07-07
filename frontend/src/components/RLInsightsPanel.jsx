import { motion } from "framer-motion";
import { Brain, Cpu, Sigma, Zap } from "lucide-react";

/**
 * RLInsightsPanel — "Glass-Box" view of the Q-Learning agent's decision.
 *
 * Every value shown is derived from the trained Q-table on the backend
 * (see rl_agent.get_action): the chosen action, its Q-value, a softmax
 * confidence over the state's Q-values, the full per-action Q-value spread,
 * and the real training hyperparameters. Nothing here is fabricated.
 *
 * Fails gracefully: renders nothing if `rlAction` is null/undefined.
 */
export default function RLInsightsPanel({ rlAction }) {
  if (!rlAction) return null;

  const {
    action = "—",
    q_value,
    confidence,
    q_values = {},
    hyperparameters = {},
  } = rlAction;

  const entries = Object.entries(q_values);
  const qNums = entries.map(([, v]) => Number(v));
  const maxQ = qNums.length ? Math.max(...qNums) : 1;
  const minQ = qNums.length ? Math.min(...qNums, 0) : 0;
  const span = maxQ - minQ || 1;

  const confidencePct =
    typeof confidence === "number" ? Math.round(confidence * 100) : null;

  return (
    <motion.div
      className="card section-gap"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      style={{
        position: "relative",
        overflow: "hidden",
        borderLeft: "4px solid var(--violet-500)",
      }}
    >
      {/* Ambient neural-interface glow */}
      <div
        aria-hidden
        style={{
          position: "absolute",
          top: -60,
          right: -60,
          width: 200,
          height: 200,
          background: "radial-gradient(circle, rgba(124,58,237,0.14), transparent 70%)",
          pointerEvents: "none",
        }}
      />

      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div>
          <p className="card-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Cpu size={12} /> Agent State — Neural Interface
          </p>
          <p className="card-title" style={{ fontSize: 16, marginBottom: 2 }}>
            Reinforcement Learning Insights
          </p>
        </div>
        <span className="badge badge-violet" style={{ whiteSpace: "nowrap" }}>
          <Brain size={11} /> Q-Learning
        </span>
      </div>

      {/* Top metric row: chosen action, Q-value, confidence, epsilon */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
          gap: 12,
          marginTop: 18,
        }}
      >
        <Metric
          label="Chosen Action"
          value={action}
          accent="var(--violet-500)"
          icon={<Zap size={13} />}
          mono
        />
        <Metric
          label="Q-Value  Q(s,a)"
          value={typeof q_value === "number" ? q_value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}
          accent="var(--blue-500)"
        />
        <Metric
          label="Confidence (softmax)"
          value={confidencePct != null ? `${confidencePct}%` : "—"}
          accent="var(--emerald-500)"
        />
        <Metric
          label="Exploration ε (start→min)"
          value={
            hyperparameters.epsilon_start != null
              ? `${hyperparameters.epsilon_start} → ${hyperparameters.epsilon_min}`
              : "—"
          }
          accent="var(--amber-500)"
        />
      </div>

      {/* Q-value spread across all actions — the real "glass box" */}
      <div style={{ marginTop: 22 }}>
        <p className="card-label" style={{ marginBottom: 10 }}>
          Q-Value Distribution Across Action Space
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {entries.map(([name, value], i) => {
            const v = Number(value);
            const pct = ((v - minQ) / span) * 100;
            const isChosen = name === action;
            return (
              <div key={name} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span
                  style={{
                    width: 150,
                    flexShrink: 0,
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    fontWeight: isChosen ? 800 : 600,
                    color: isChosen ? "var(--violet-500)" : "var(--text-muted)",
                  }}
                >
                  {name}
                </span>
                <div
                  style={{
                    flex: 1,
                    height: 18,
                    background: "var(--bg-card-hover)",
                    borderRadius: 6,
                    overflow: "hidden",
                    border: "1px solid var(--border)",
                  }}
                >
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(2, pct)}%` }}
                    transition={{ duration: 0.6, delay: 0.1 + i * 0.06, ease: "easeOut" }}
                    style={{
                      height: "100%",
                      borderRadius: 6,
                      background: isChosen
                        ? "linear-gradient(90deg, #7c3aed, #8b5cf6)"
                        : "linear-gradient(90deg, #94a3b8, #cbd5e1)",
                    }}
                  />
                </div>
                <span
                  style={{
                    width: 78,
                    textAlign: "right",
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    fontWeight: 700,
                    color: isChosen ? "var(--text-primary)" : "var(--text-muted)",
                  }}
                >
                  {v.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Bellman equation badge */}
      <div
        style={{
          marginTop: 22,
          padding: "14px 16px",
          borderRadius: "var(--r-md)",
          background: "linear-gradient(135deg, rgba(124,58,237,0.08), rgba(37,99,235,0.05))",
          border: "1px solid rgba(124,58,237,0.2)",
        }}
      >
        <p
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 10,
            fontWeight: 800,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--violet-500)",
            marginBottom: 8,
          }}
        >
          <Sigma size={12} /> Bellman Optimality — Governing Update Rule
        </p>
        <code
          style={{
            display: "block",
            fontFamily: "var(--font-mono)",
            fontSize: 13.5,
            fontWeight: 700,
            color: "var(--text-primary)",
            letterSpacing: "0.01em",
          }}
        >
          Q(s,a) = R(s,a) + γ · max<sub>a'</sub> Q(s',a')
        </code>
        <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8, lineHeight: 1.5 }}>
          Trained via temporal-difference learning with discount factor{" "}
          <strong style={{ color: "var(--text-secondary)" }}>
            γ = {hyperparameters.gamma ?? "0.95"}
          </strong>
          . {hyperparameters.policy || "greedy (argmax) at inference"}.
        </p>
      </div>
    </motion.div>
  );
}

function Metric({ label, value, accent, icon, mono }) {
  return (
    <div
      style={{
        padding: "12px 14px",
        borderRadius: "var(--r-md)",
        background: "var(--bg-card-hover)",
        border: "1px solid var(--border)",
      }}
    >
      <p
        className="card-label"
        style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 6 }}
      >
        {icon}
        {label}
      </p>
      <p
        style={{
          fontSize: mono ? 14 : 18,
          fontWeight: 800,
          color: accent,
          fontFamily: mono ? "var(--font-mono)" : "var(--font-display)",
          wordBreak: "break-word",
        }}
      >
        {value}
      </p>
    </div>
  );
}
