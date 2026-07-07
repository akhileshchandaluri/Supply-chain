import { useEffect, useState } from "react";
import { motion, AnimatePresence, useSpring, useTransform } from "framer-motion";
import { BrainCircuit, Scale, Sigma } from "lucide-react";
import RLInsightsPanel from "../components/RLInsightsPanel";
import RLChatPanel from "../components/RLChatPanel";

// Human-readable labels for the reward-function components (mirror rl_agent.explain_reward).
const REWARD_LABELS = {
  holding_cost: "Holding cost",
  demand_fulfillment: "Demand fulfilled",
  stockout_penalty: "Stockout penalty",
  overcapacity_penalty: "Overcapacity penalty",
  action_cost: "Action cost",
  anomaly_mitigation: "Anomaly mitigation",
};

// "Action chosen because X (long-term value) beat the next-best alternative Y."
function buildJustification(rl) {
  if (!rl?.q_values) return null;
  const sorted = Object.entries(rl.q_values).sort(([, a], [, b]) => b - a);
  const [chosenName, chosenQ] = sorted[0];
  const runnerUp = sorted[1];
  const rb = rl.reward_breakdown?.components || {};
  const positives = Object.entries(rb).filter(([, v]) => v > 0).sort(([, a], [, b]) => b - a);
  const costs = Object.entries(rb).filter(([, v]) => v < 0).sort(([, a], [, b]) => a - b);
  return {
    chosenName,
    chosenQ,
    runnerUpName: runnerUp?.[0],
    runnerUpQ: runnerUp?.[1],
    topReward: positives[0],
    topCost: costs[0],
    total: rl.reward_breakdown?.total,
  };
}

function RewardBreakdown({ rl }) {
  const rb = rl?.reward_breakdown;
  if (!rb?.components) return null;
  const rows = Object.entries(rb.components);
  return (
    <div className="card">
      <p className="card-label" style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
        <Sigma size={12} /> Reward / Points Breakdown
      </p>
      <p className="card-subtitle" style={{ marginBottom: 14 }}>
        One-step reward decomposition for the chosen action (from the trained reward function).
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {rows.map(([key, val]) => {
          const positive = val >= 0;
          return (
            <div
              key={key}
              style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "10px 14px", borderRadius: "var(--r-sm)",
                background: "var(--bg-card-hover)", border: "1px solid var(--border)",
              }}
            >
              <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-secondary)" }}>
                {REWARD_LABELS[key] || key}
              </span>
              <span
                style={{
                  fontFamily: "var(--font-mono)", fontSize: 14, fontWeight: 800,
                  color: positive ? "var(--emerald-500)" : "var(--rose-500)",
                }}
              >
                {positive ? "+" : ""}{val}
              </span>
            </div>
          );
        })}
        <div
          style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "12px 14px", borderRadius: "var(--r-sm)", marginTop: 4,
            background: "linear-gradient(135deg, rgba(37,99,235,0.08), rgba(124,58,237,0.05))",
            border: "1px solid var(--border-bright)",
          }}
        >
          <span style={{ fontSize: 12, fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
            Total Step Reward
          </span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 16, fontWeight: 800, color: "var(--blue-600)" }}>
            {rb.total}
          </span>
        </div>
      </div>
    </div>
  );
}

function JustificationCard({ rl }) {
  const j = buildJustification(rl);
  if (!j) return null;
  const fmt = (n) => (typeof n === "number" ? n.toLocaleString(undefined, { maximumFractionDigits: 1 }) : n);
  return (
    <div className="card" style={{ borderLeft: "4px solid var(--violet-500)" }}>
      <p className="card-label" style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
        <Scale size={12} /> Decision Justification
      </p>
      <p style={{ fontSize: 14, lineHeight: 1.7, color: "var(--text-secondary)" }}>
        <strong style={{ color: "var(--text-primary)" }}>{j.chosenName.replace(/_/g, " ")}</strong> was chosen
        because its expected long-term value{" "}
        <strong style={{ fontFamily: "var(--font-mono)", color: "var(--violet-500)" }}>Q = {fmt(j.chosenQ)}</strong>
        {j.runnerUpName && (
          <> exceeded the next-best option{" "}
            <strong style={{ color: "var(--text-primary)" }}>{j.runnerUpName.replace(/_/g, " ")}</strong>{" "}
            (<span style={{ fontFamily: "var(--font-mono)" }}>Q = {fmt(j.runnerUpQ)}</span>)</>
        )}
        .{" "}
        {j.topReward && (
          <>Immediate reward is driven mainly by{" "}
            <strong style={{ color: "var(--emerald-500)" }}>{REWARD_LABELS[j.topReward[0]] || j.topReward[0]} (+{j.topReward[1]})</strong>
            {j.topCost && (
              <>, outweighing the{" "}
                <strong style={{ color: "var(--rose-500)" }}>{REWARD_LABELS[j.topCost[0]] || j.topCost[0]} ({j.topCost[1]})</strong></>
            )}
            , for a net step reward of{" "}
            <strong style={{ fontFamily: "var(--font-mono)", color: "var(--blue-600)" }}>{fmt(j.total)}</strong>.
          </>
        )}
      </p>
    </div>
  );
}

// ─── Action configuration (mirrors rl_agent.py ACTIONS dict) ──────────────────
const ACTION_CFG = {
  HOLD:              { emoji:"⏸️", label:"Hold",             color:"#6366F1", bg:"rgba(99,102,241,0.08)", border:"rgba(99,102,241,0.3)", glow:"rgba(99,102,241,0.15)", qty:"+0",    desc:"Inventory is in the optimal zone. No reorder needed." },
  REORDER_SMALL:     { emoji:"📦", label:"Reorder Small",    color:"#10B981", bg:"rgba(16,185,129,0.08)", border:"rgba(16,185,129,0.3)", glow:"rgba(16,185,129,0.15)", qty:"+50",   desc:"Minor shortfall ahead. Order 50 units to top up." },
  REORDER_MEDIUM:    { emoji:"🚛", label:"Reorder Medium",   color:"#F59E0B", bg:"rgba(245,158,11,0.08)", border:"rgba(245,158,11,0.3)", glow:"rgba(245,158,11,0.15)", qty:"+150",  desc:"Moderate restocking needed — 150 units inbound." },
  REORDER_LARGE:     { emoji:"🏭", label:"Reorder Large",    color:"#FB923C", bg:"rgba(251,146,60,0.08)", border:"rgba(251,146,60,0.3)", glow:"rgba(251,146,60,0.15)", qty:"+300",  desc:"Significant replenishment: 300 units ordered." },
  EMERGENCY_REORDER: { emoji:"🚨", label:"Emergency Reorder",color:"#F43F5E", bg:"rgba(244,63,94,0.08)",  border:"rgba(244,63,94,0.3)",  glow:"rgba(244,63,94,0.15)",  qty:"+200",  desc:"Critical shortage! Auto-activated emergency route." },
  SWITCH_SUPPLIER:   { emoji:"⚡", label:"Switch Supplier",  color:"#A78BFA", bg:"rgba(167,139,250,0.08)",border:"rgba(167,139,250,0.3)", glow:"rgba(167,139,250,0.15)", qty:"0",    desc:"Supplier anomaly detected — switching to alternate vendor." },
};

function WarningBanner({ action }) {
  const isEmergency = action === "EMERGENCY_REORDER";
  const isSwitch    = action === "SWITCH_SUPPLIER";
  if (!isEmergency && !isSwitch) return null;

  return (
    <AnimatePresence>
      <motion.div
        key={action}
        initial={{ opacity: 0, y: -20, scale: 0.97 }}
        animate={{ opacity: 1, y: 0,  scale: 1 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ type: "spring", stiffness: 300, damping: 24 }}
        style={{
          marginBottom: 20,
          padding: "14px 20px",
          borderRadius: 14,
          border: `1px solid ${isEmergency ? "rgba(244,63,94,0.3)" : "rgba(167,139,250,0.3)"}`,
          background: isEmergency ? "rgba(244,63,94,0.05)" : "rgba(167,139,250,0.05)",
          display: "flex", alignItems: "center", gap: 14,
          boxShadow: isEmergency
            ? "0 4px 12px rgba(244,63,94,0.05)"
            : "0 4px 12px rgba(167,139,250,0.05)",
        }}
      >
        <motion.div
          animate={{ scale: [1, 1.18, 1], rotate: isEmergency ? [0, -5, 5, 0] : 0 }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
          style={{ fontSize: 28, lineHeight: 1 }}
        >
          {isEmergency ? "🚨" : "⚡"}
        </motion.div>

        <div style={{ flex: 1 }}>
          <p style={{
            fontFamily: "var(--font-display)", fontSize: 15, fontWeight: 800,
            color: isEmergency ? "#E11D48" : "#7C3AED", marginBottom: 3,
          }}>
            {isEmergency ? "CRITICAL: Emergency Reorder Triggered" : "WARNING: Supplier Anomaly — Switching Vendor"}
          </p>
          <p style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.5 }}>
            {isEmergency
              ? "Q-agent detected imminent stockout. A* emergency route auto-activated (cost penalty accepted)."
              : "Isolation Forest flagged supplier anomaly. Q-agent overrides standard order."}
          </p>
        </div>

        <div style={{
          padding: "5px 12px", borderRadius: 999,
          background: isEmergency ? "rgba(244,63,94,0.1)" : "rgba(167,139,250,0.1)",
          border: `1px solid ${isEmergency ? "rgba(244,63,94,0.2)" : "rgba(167,139,250,0.2)"}`,
          fontSize: 10, fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.1em",
          color: isEmergency ? "#E11D48" : "#7C3AED", flexShrink: 0,
        }}>
          {isEmergency ? "SEVERITY: HIGH" : "SEVERITY: MED"}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

function CountUp({ value, decimals = 3 }) {
  const spring = useSpring(value, { stiffness: 80, damping: 18 });
  const display = useTransform(spring, (v) => v.toFixed(decimals));
  const [text, setText] = useState(value.toFixed(decimals));

  useEffect(() => {
    spring.set(value);
    const unsub = display.on("change", (v) => setText(v));
    return unsub;
  }, [display, spring, value]);

  return <span>{text}</span>;
}

function QCard({ actionName, qval, isChosen, rank, delay }) {
  const cfg = ACTION_CFG[actionName] || ACTION_CFG.HOLD;
  const barPct = Math.max(0, Math.min(100, (qval + 20) * 2.5));

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.92 }}
      animate={{ opacity: 1, y: 0,  scale: 1 }}
      transition={{ delay, type: "spring", stiffness: 260, damping: 20 }}
      style={{
        padding: "14px 16px",
        borderRadius: 14,
        background: isChosen ? cfg.bg : "var(--bg-card)",
        border: `1px solid ${isChosen ? cfg.border : "var(--border)"}`,
        boxShadow: isChosen ? `0 4px 15px ${cfg.glow}` : "none",
        transition: "all 0.25s ease",
        position: "relative", overflow: "hidden",
      }}
    >
      {isChosen && (
        <motion.div
          initial={{ scale: 0, rotate: -20 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: "spring", stiffness: 400, damping: 15 }}
          style={{
            position: "absolute", top: 8, right: 10,
            fontSize: 11, fontWeight: 800, color: cfg.color,
            background: `${cfg.color}18`,
            border: `1px solid ${cfg.color}44`,
            padding: "2px 7px", borderRadius: 999,
            letterSpacing: "0.06em",
          }}
        >
          CHOSEN
        </motion.div>
      )}

      {rank <= 3 && !isChosen && (
        <div style={{
          position: "absolute", top: 8, right: 10,
          fontSize: 10, color: "var(--text-dim)",
        }}>#{rank}</div>
      )}

      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 10 }}>
        <span style={{ fontSize: 17 }}>{cfg.emoji}</span>
        <div>
          <p style={{ fontSize: 9, fontWeight: 800, color: isChosen ? cfg.color : "var(--text-muted)",
            textTransform: "uppercase", letterSpacing: "0.08em" }}>
            {actionName.replace(/_/g, " ")}
          </p>
          <p style={{ fontSize: 9, color: "var(--text-dim)", marginTop: 1 }}>qty {cfg.qty}</p>
        </div>
      </div>

      <div style={{
        fontFamily: "var(--font-mono)",
        fontSize: 22, fontWeight: 800, lineHeight: 1,
        color: isChosen ? cfg.color : "var(--text-primary)",
        marginBottom: 8,
      }}>
        <CountUp value={qval} decimals={3} />
      </div>

      <div style={{
        height: 3, borderRadius: 999,
        background: "rgba(0,0,0,0.05)", overflow: "hidden",
      }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${barPct}%` }}
          transition={{ duration: 0.7, delay: delay + 0.1, ease: [0.4, 0, 0.2, 1] }}
          style={{
            height: "100%", borderRadius: 999,
            background: isChosen
              ? `linear-gradient(90deg, ${cfg.color}, ${cfg.color}88)`
              : "rgba(0,0,0,0.1)",
          }}
        />
      </div>
    </motion.div>
  );
}

function HeroCard({ result }) {
  if (!result) {
    return (
      <motion.div
        initial={{ opacity:0 }} animate={{ opacity:1 }}
        style={{
          display:"flex", flexDirection:"column",
          alignItems:"center", justifyContent:"center", gap:16,
          padding:60, border:"1px dashed var(--border)",
          borderRadius:18, background:"var(--bg-card)", textAlign:"center",
        }}
      >
        <motion.div
          animate={{ scale:[1,1.08,1], rotate:[0,5,-5,0] }}
          transition={{ duration:3, repeat:Infinity, ease:"easeInOut" }}
          style={{ fontSize:60 }}
        >🤖</motion.div>
        <div>
          <p style={{ fontSize:15, fontWeight:700, color:"var(--text-secondary)", marginBottom:6 }}>
            Awaiting live order simulation
          </p>
          <p style={{ fontSize:12, color:"var(--text-muted)", lineHeight:1.5 }}>
            The Q-agent provides inventory recommendations automatically
          </p>
        </div>
      </motion.div>
    );
  }

  const cfg = ACTION_CFG[result.action] || ACTION_CFG.HOLD;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={result.action}
        initial={{ opacity:0, scale:0.88, y:16 }}
        animate={{ opacity:1, scale:1,   y:0  }}
        exit={{   opacity:0, scale:0.92, y:-8 }}
        transition={{ type:"spring", stiffness:260, damping:20 }}
        style={{
          padding: "40px 24px",
          borderRadius: 20,
          background: cfg.bg,
          border: `1.5px solid ${cfg.border}`,
          boxShadow: `0 8px 30px ${cfg.glow}`,
          textAlign: "center",
        }}
      >
        <motion.div
          animate={{ scale:[1,1.1,1] }}
          transition={{ duration:2, repeat:Infinity, ease:"easeInOut" }}
          style={{ fontSize:62, marginBottom:12, lineHeight:1 }}
        >
          {cfg.emoji}
        </motion.div>

        <p style={{
          fontFamily:"var(--font-display)",
          fontSize:28, fontWeight:900,
          color:cfg.color, letterSpacing:"-0.03em",
          marginBottom:4,
        }}>
          {result.action.replace(/_/g," ")}
        </p>

        <div style={{
          display:"inline-flex", alignItems:"center", gap:5,
          padding:"6px 16px", borderRadius:999, margin:"12px 0",
          background:`${cfg.color}15`, border:`1px solid ${cfg.color}35`,
          fontSize:13, fontWeight:700, color:cfg.color,
        }}>
          Reorder: {cfg.qty} units
        </div>

        <p style={{ fontSize:14, color:"var(--text-secondary)", lineHeight:1.6, marginTop:6 }}>
          {cfg.desc}
        </p>

        <div style={{
          marginTop:24, padding:"12px 16px", borderRadius:10,
          background:"#FFFFFF", border:"1px solid var(--border)",
          fontFamily:"var(--font-mono)", fontSize:12, color:"var(--text-muted)",
          textAlign:"left", display: "inline-block"
        }}>
          <span style={{ color:"var(--text-dim)", marginRight:8 }}>Q-table lookup:</span>
          <span style={{ color:cfg.color }}>argmax Q[state]</span>
          <span style={{ color:"var(--text-secondary)" }}> = {result.action_id} → {result.action}</span>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

export default function RLAgent({ liveOrder }) {
  const result = liveOrder?.rl_action;

  const qRanks = result?.q_values
    ? Object.entries(result.q_values)
        .sort(([, a], [, b]) => b - a)
        .reduce((acc, [name], i) => { acc[name] = i + 1; return acc; }, {})
    : {};

  const isAlert = result?.action === "EMERGENCY_REORDER" || result?.action === "SWITCH_SUPPLIER";

  return (
    <motion.div initial={{ opacity:0, y:12 }} animate={{ opacity:1, y:0 }} transition={{ duration:0.35 }}>
      <div className="page-header">
        <div className="page-badge"><BrainCircuit size={10} /> Q-Learning · Tabular RL</div>
        <h1 className="page-title">RL Inventory Agent</h1>
        <p className="page-subtitle">Interactive expert system: the Q-Learning decision, its mathematical justification, reward breakdown, and an AI analyst you can question.</p>
      </div>

      {isAlert && <WarningBanner action={result.action} />}

      <div style={{ maxWidth: 860, margin: "0 auto", display:"flex", flexDirection:"column", gap:24 }}>
        <div className="card">
          <p className="card-label" style={{ marginBottom:10 }}>Agent Decision</p>
          <HeroCard result={result} />
        </div>

        <AnimatePresence>
          {result?.q_values && (
            <motion.div className="card" initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}>
              <p className="card-label" style={{ marginBottom:16 }}>
                Q-Values — All 6 Actions
                <span style={{ float:"right", fontSize:10, color:"var(--text-dim)", fontWeight:500 }}>
                  current state lookup
                </span>
              </p>
              <div style={{
                display:"grid",
                gridTemplateColumns:"repeat(auto-fit, minmax(220px, 1fr))",
                gap:12,
              }}>
                {Object.entries(result.q_values).map(([actionName, qval], i) => (
                  <QCard
                    key={actionName}
                    actionName={actionName}
                    qval={qval}
                    isChosen={actionName === result.action}
                    rank={qRanks[actionName] ?? i+1}
                    delay={i * 0.05}
                  />
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Mathematical justification: why this action beat the alternatives */}
        {result && <JustificationCard rl={result} />}

        {/* Itemized reward / points breakdown */}
        {result && <RewardBreakdown rl={result} />}

        {/* Glass-box RL internals (Q-value spread, Bellman badge, confidence) */}
        <RLInsightsPanel rlAction={result} />

        {/* Interactive, context-aware LLM analyst (replaces the static audit log) */}
        <RLChatPanel liveOrder={liveOrder} />
      </div>
    </motion.div>
  );
}
