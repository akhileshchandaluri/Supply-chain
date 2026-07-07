import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Send, Sparkles, Loader2, Terminal } from "lucide-react";
import { chatRL } from "../api/client";

/**
 * RLChatPanel — interactive, context-aware LLM chat about the latest RL decision.
 *
 * Replaces the old static "AI Core" audit log. It sends the current run's real
 * decision data (Q-values, reward breakdown, optimization, risk) to /api/rl/chat
 * so the LLM grounds every answer in the actual pipeline output. Seeds the
 * conversation with the one-shot executive audit (ai_explanation).
 */
const SUGGESTIONS = [
  "Why was this action chosen?",
  "Why not a large reorder?",
  "What if the risk score were higher?",
  "Explain the reward breakdown.",
];

export default function RLChatPanel({ liveOrder }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  const rl = liveOrder?.rl_action;

  // Seed with the executive audit whenever a new run arrives.
  useEffect(() => {
    if (!rl) {
      setMessages([]);
      return;
    }
    const seed = liveOrder?.ai_explanation;
    setMessages(
      seed
        ? [{ role: "assistant", content: seed }]
        : [{ role: "assistant", content: "Decision data loaded. Ask me anything about why the agent chose this action." }]
    );
  }, [rl, liveOrder?.ai_explanation]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  // Compact, grounded context sent with every question.
  const buildContext = () => ({
    action: rl?.action,
    q_value: rl?.q_value,
    confidence: rl?.confidence,
    q_values: rl?.q_values,
    reward_breakdown: rl?.reward_breakdown,
    hyperparameters: rl?.hyperparameters,
    risk_level: liveOrder?.risk_level,
    risk_score: liveOrder?.risk_score,
    anomaly: liveOrder?.anomaly,
    demand_7d_avg: liveOrder?.demand_7d_avg,
    is_emergency: liveOrder?.is_emergency,
    optimization: liveOrder?.optimization
      ? {
          status: liveOrder.optimization.status,
          total_optimized_cost: liveOrder.optimization.total_optimized_cost,
          allocation_count: liveOrder.optimization.allocations?.length ?? 0,
        }
      : null,
  });

  const send = async (question) => {
    const q = (question ?? input).trim();
    if (!q || loading || !rl) return;
    setInput("");
    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await chatRL({ question: q, context: buildContext(), history });
      setMessages((prev) => [...prev, { role: "assistant", content: res.data.answer }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "⚠️ Chat unavailable — check the backend and GROQ_API_KEY." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const disabled = !rl;

  return (
    <div className="card" style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
      {/* Terminal-style header */}
      <div
        style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "12px 18px", background: "#0f172a",
          borderBottom: "1px solid rgba(148,163,184,0.18)",
        }}
      >
        <span style={{ display: "flex", gap: 6 }}>
          <span style={{ width: 11, height: 11, borderRadius: "50%", background: "#f43f5e" }} />
          <span style={{ width: 11, height: 11, borderRadius: "50%", background: "#f59e0b" }} />
          <span style={{ width: 11, height: 11, borderRadius: "50%", background: "#10b981" }} />
        </span>
        <Terminal size={14} style={{ color: "#38bdf8", marginLeft: 6 }} />
        <span style={{ color: "#e2e8f0", fontSize: 12, fontWeight: 800, letterSpacing: "0.06em", fontFamily: "var(--font-mono)" }}>
          AI CORE — INTERACTIVE ANALYST
        </span>
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 5, color: "#64748b", fontSize: 10, fontWeight: 700 }}>
          <Sparkles size={11} /> Llama 3.3 · Groq
        </span>
      </div>

      {/* Message stream */}
      <div
        ref={scrollRef}
        style={{
          background: "#0b1120", padding: "18px 20px",
          height: 300, overflowY: "auto",
          display: "flex", flexDirection: "column", gap: 12,
        }}
      >
        {disabled ? (
          <div style={{ margin: "auto", textAlign: "center", color: "#64748b", fontSize: 13 }}>
            <Terminal size={28} style={{ marginBottom: 10, opacity: 0.6 }} />
            <p>Run a pipeline from the Overview tab to start the analysis.</p>
          </div>
        ) : (
          messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}
            >
              <div
                style={{
                  maxWidth: "82%",
                  padding: "10px 14px",
                  borderRadius: 12,
                  fontSize: 13,
                  lineHeight: 1.6,
                  whiteSpace: "pre-wrap",
                  fontFamily: m.role === "assistant" ? "var(--font-mono)" : "var(--font-body)",
                  background: m.role === "user" ? "#2563eb" : "rgba(148,163,184,0.1)",
                  color: m.role === "user" ? "#fff" : "#a5f3fc",
                  border: m.role === "assistant" ? "1px solid rgba(56,189,248,0.15)" : "none",
                }}
              >
                {m.content}
              </div>
            </motion.div>
          ))
        )}
        {loading && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: "#64748b", fontSize: 12, fontFamily: "var(--font-mono)" }}>
            <Loader2 size={13} className="spin" /> analyzing Q-table…
          </div>
        )}
      </div>

      {/* Suggestion chips */}
      {!disabled && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "10px 14px 0" }}>
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              disabled={loading}
              style={{
                fontSize: 11, fontWeight: 600, padding: "5px 10px", borderRadius: 999,
                border: "1px solid var(--border)", background: "var(--bg-card-hover)",
                color: "var(--text-secondary)", cursor: loading ? "default" : "pointer",
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{ display: "flex", gap: 8, padding: 14 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          disabled={disabled || loading}
          placeholder={disabled ? "Awaiting a pipeline run…" : "Ask why the agent made this decision…"}
          style={{
            flex: 1, padding: "10px 14px", borderRadius: "var(--r-sm)",
            border: "1px solid var(--border-strong)", background: "var(--bg-card)",
            color: "var(--text-primary)", fontSize: 13, outline: "none",
          }}
        />
        <motion.button
          className="btn btn-primary"
          onClick={() => send()}
          disabled={disabled || loading || !input.trim()}
          whileTap={{ scale: 0.96 }}
        >
          {loading ? <Loader2 size={15} className="spin" /> : <Send size={15} />}
        </motion.button>
      </div>
    </div>
  );
}
