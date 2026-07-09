import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { getStatus, runPipeline } from "../api/client";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  BarChart3,
  Brain,
  CheckCircle2,
  Clock,
  Database,
  History,
  Loader2,
  Network,
  Package,
  Rocket,
  ScanSearch,
  Settings,
  SlidersHorizontal,
  TrendingUp,
  X,
  Zap,
} from "lucide-react";

const PIPELINE = [
  { icon: Database,    label: "DataCo CSV",    sub: "Order history",    color: "linear-gradient(135deg,#2563eb,#38bdf8)" },
  { icon: Settings,    label: "Feature Eng.",   sub: "Lag and rolling sets", color: "linear-gradient(135deg,#0d9488,#2dd4bf)" },
  { icon: TrendingUp,  label: "XGBoost",        sub: "Demand forecast",  color: "linear-gradient(135deg,#06b6d4,#2563eb)" },
  { icon: BarChart3,   label: "Random Forest",  sub: "Risk classes",     color: "linear-gradient(135deg,#10b981,#0d9488)" },
  { icon: Brain,       label: "Q-Learning",     sub: "Inventory action", color: "linear-gradient(135deg,#f43f5e,#8b5cf6)" },
  { icon: History,     label: "History",         sub: "Decision audit",   color: "linear-gradient(135deg,#64748b,#2563eb)" },
  { icon: Network,     label: "Routing",         sub: "A* shortest-path", color: "linear-gradient(135deg,#7c3aed,#2563eb)" },
];

const HERO_KPIS = [
  { value: "6", label: "AI modules" },
  { value: "18", label: "Network cities" },
  { value: "4", label: "Decision models" },
];

function NetworkGraphic() {
  const nodes = [
    { x: 62, y: 70, r: 7, color: "#2563eb" },
    { x: 168, y: 48, r: 10, color: "#0d9488" },
    { x: 282, y: 92, r: 7, color: "#f59e0b" },
    { x: 94, y: 190, r: 9, color: "#10b981" },
    { x: 214, y: 178, r: 12, color: "#7c3aed" },
    { x: 332, y: 224, r: 8, color: "#f43f5e" },
    { x: 148, y: 292, r: 8, color: "#06b6d4" },
    { x: 284, y: 316, r: 10, color: "#2563eb" },
  ];

  const paths = [
    "M62 70 C104 44 130 42 168 48 S246 74 282 92",
    "M62 70 C82 124 78 160 94 190 S132 258 148 292",
    "M168 48 C178 94 188 132 214 178 S284 210 332 224",
    "M94 190 C134 158 170 158 214 178 S260 250 284 316",
    "M148 292 C192 306 238 326 284 316",
  ];

  return (
    <svg className="network-graphic" viewBox="0 0 400 360" role="img" aria-label="Animated supply chain network">
      <defs>
        <linearGradient id="networkStroke" x1="0" x2="1" y1="0" y2="1">
          <stop stopColor="#2563eb" />
          <stop offset="0.52" stopColor="#0d9488" />
          <stop offset="1" stopColor="#f59e0b" />
        </linearGradient>
      </defs>

      <g opacity="0.42">
        {[40, 90, 140, 190, 240, 290, 340].map((x) => (
          <line key={`v-${x}`} x1={x} y1="30" x2={x} y2="330" stroke="#94a3b8" strokeWidth="0.6" opacity="0.22" />
        ))}
        {[56, 110, 164, 218, 272, 326].map((y) => (
          <line key={`h-${y}`} x1="34" y1={y} x2="366" y2={y} stroke="#94a3b8" strokeWidth="0.6" opacity="0.22" />
        ))}
      </g>

      {paths.map((d, i) => (
        <motion.path
          key={d}
          d={d}
          fill="none"
          stroke="url(#networkStroke)"
          strokeWidth="4"
          strokeLinecap="round"
          opacity="0.22"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, delay: i * 0.08, ease: "easeOut" }}
        />
      ))}

      {paths.map((d, i) => (
        <path
          key={`flow-${d}`}
          className="data-path"
          d={d}
          fill="none"
          stroke={i % 2 ? "#0d9488" : "#2563eb"}
          strokeWidth="2"
          strokeLinecap="round"
          opacity="0.65"
        />
      ))}

      {nodes.map((node, i) => (
        <motion.g key={`${node.x}-${node.y}`} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.25 + i * 0.06, type: "spring", stiffness: 260, damping: 18 }}>
          <motion.circle
            cx={node.x}
            cy={node.y}
            r={node.r + 9}
            fill={node.color}
            opacity="0.1"
            animate={{ scale: [1, 1.35, 1], opacity: [0.12, 0.04, 0.12] }}
            transition={{ duration: 2.8, repeat: Infinity, delay: i * 0.18, ease: "easeInOut" }}
          />
          <circle cx={node.x} cy={node.y} r={node.r} fill={node.color} stroke="#fff" strokeWidth="3" />
        </motion.g>
      ))}

      <motion.g initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.75 }}>
        <rect x="78" y="24" width="190" height="34" rx="8" fill="#ffffff" stroke="rgba(15,23,42,0.08)" />
        <text x="94" y="46" fill="#475569" fontSize="12" fontWeight="800" fontFamily="Inter, sans-serif">
          Live model signal flow
        </text>
      </motion.g>
    </svg>
  );
}

// ─── Command Center config ────────────────────────────────────────────────────
// Sequential pipeline stages shown in the stepper. `key` maps to a payload the
// backend understands; the labels are demo-facing.
const PIPELINE_STAGES = [
  { id: "forecast",   label: "Forecasting",        icon: TrendingUp },
  { id: "risk",       label: "Risk Assessment",    icon: BarChart3 },
  { id: "rl",         label: "RL Strategy",        icon: Brain },
  { id: "ortools",    label: "OR-Tools Allocation", icon: Package },
  { id: "audit",      label: "AI Auditing",        icon: ScanSearch },
];

// Destination hub label → logistics graph node ID (see graph_construction.py).
const HUB_NODES = {
  Bengaluru: 6,
  Mumbai: 0,
  Chennai: 2,
  Delhi: 1,
};

// Delivery urgency → days_to_delivery + shipping mode encoding for the payload.
const URGENCY_PRESETS = {
  Standard:  { days_to_delivery: 6, shipping_mode_enc: 0 },
  Expedited: { days_to_delivery: 3, shipping_mode_enc: 1 },
  Emergency: { days_to_delivery: 1, shipping_mode_enc: 2 },
};

// Source warehouse label → logistics graph node ID (routing origin / start_node).
const SOURCE_NODES = {
  Mumbai: 0,
  Delhi: 1,
  Chennai: 2,
  Kochi: 11,
  Guwahati: 17,
};

// Supplier reliability → Isolation Forest metrics + RF supplier_delay_rate.
// "Low" is tuned to look anomalous (poor fulfillment, slow, many complaints).
const RELIABILITY_PRESETS = {
  High:   { fulfillment_rate: 0.95, avg_delivery_time: 3.0, complaint_freq: 3.0,  price_deviation: 0.05, supplier_delay_rate: 0.10 },
  Medium: { fulfillment_rate: 0.85, avg_delivery_time: 5.0, complaint_freq: 10.0, price_deviation: 0.10, supplier_delay_rate: 0.20 },
  Low:    { fulfillment_rate: 0.55, avg_delivery_time: 9.0, complaint_freq: 28.0, price_deviation: 0.40, supplier_delay_rate: 0.45 },
};

// Shared input/select styling consistent with the design system.
const fieldStyle = {
  padding: "9px 12px",
  border: "1px solid var(--border-strong)",
  borderRadius: "var(--r-sm)",
  background: "var(--bg-card)",
  color: "var(--text-primary)",
  fontSize: 13,
  fontWeight: 600,
  fontFamily: "var(--font-body)",
  outline: "none",
};

// Build the PipelineRequest body from Command Center inputs.
function buildPayload({ quantity, urgency, hub, inventory, reliability, source, orderValue, discountRate }) {
  const preset = URGENCY_PRESETS[urgency] || URGENCY_PRESETS.Standard;
  const rel = RELIABILITY_PRESETS[reliability] || RELIABILITY_PRESETS.Medium;
  const goalNode = HUB_NODES[hub] ?? 6;
  // Start and goal must differ; if the source equals the destination, nudge start.
  let startNode = SOURCE_NODES[source] ?? 0;
  if (startNode === goalNode) startNode = goalNode === 0 ? 1 : 0;

  // Order value defaults to quantity·100 if not explicitly provided.
  const value = Number(orderValue) > 0
    ? Number(orderValue)
    : Math.max(1, Number(quantity) || 1) * 100.0;

  return {
    inventory: Math.max(0, Number(inventory) >= 0 ? Number(inventory) : 120),
    days_to_delivery: preset.days_to_delivery,
    start_node: startNode,
    goal_node: goalNode,
    order_quantity: Math.max(1, Number(quantity) || 1), // scales the demand forecast
    risk_features: {
      shipping_mode_enc: preset.shipping_mode_enc,
      actual_days: preset.days_to_delivery + 1,
      scheduled_days: preset.days_to_delivery,
      discount_rate: Math.min(1, Math.max(0, Number(discountRate) >= 0 ? Number(discountRate) : 0.1)),
      order_value: value,
      supplier_delay_rate: rel.supplier_delay_rate,
      days_buffer: 0.0,
      delay_gap: 1.0,
    },
    supplier_metrics: {
      avg_delivery_time: rel.avg_delivery_time,
      price_deviation: rel.price_deviation,
      fulfillment_rate: rel.fulfillment_rate,
      complaint_freq: rel.complaint_freq,
    },
  };
}

// The visual stepper. `activeIndex` = stage currently running; anything before is
// complete, anything after is pending. `done` marks the whole run finished.
function PipelineStepper({ activeIndex, done }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 16 }}>
      {PIPELINE_STAGES.map((stage, i) => {
        const Icon = stage.icon;
        const state =
          done || i < activeIndex ? "complete" : i === activeIndex ? "active" : "pending";
        const palette = {
          complete: { bg: "rgba(16,185,129,0.09)", border: "rgba(16,185,129,0.28)", color: "#047857" },
          active:   { bg: "rgba(37,99,235,0.1)",   border: "rgba(37,99,235,0.35)",  color: "var(--blue-600)" },
          pending:  { bg: "#f8fafc",                border: "var(--border)",         color: "var(--text-dim)" },
        }[state];
        return (
          <motion.div
            key={stage.id}
            initial={false}
            animate={{
              backgroundColor: palette.bg,
              borderColor: palette.border,
              scale: state === "active" ? 1.03 : 1,
            }}
            transition={{ duration: 0.3 }}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 12px", borderRadius: 999,
              border: `1px solid ${palette.border}`,
              fontSize: 12, fontWeight: 800, color: palette.color,
            }}
          >
            {state === "complete" ? (
              <CheckCircle2 size={14} />
            ) : state === "active" ? (
              <Loader2 size={14} className="spin" />
            ) : (
              <Icon size={14} />
            )}
            {stage.label}
          </motion.div>
        );
      })}
    </div>
  );
}

export default function Dashboard({ onNavigate, setLiveOrder }) {
  const [status, setStatus] = useState(null);

  // ─── Command Center state ───────────────────────────────────────────────────
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({
    quantity: 50,
    urgency: "Standard",
    hub: "Bengaluru",
    inventory: 120,
    reliability: "Medium",
    source: "Mumbai",
    orderValue: 500,
    discountRate: 0.1,
  });
  const [running, setRunning] = useState(false);
  const [activeStage, setActiveStage] = useState(-1);
  const [runComplete, setRunComplete] = useState(false);
  const [result, setResult] = useState(null);
  const [runError, setRunError] = useState(null);
  const stepTimers = useRef([]);

  // Drive the stepper animation while the API request is in flight. The stages
  // advance on a timer for visual feedback; the real result lands independently.
  const startStepperAnimation = () => {
    stepTimers.current.forEach(clearTimeout);
    stepTimers.current = [];
    setActiveStage(0);
    // Advance through the first N-1 stages; the final "AI Auditing" stage stays
    // active until the real response resolves and marks the run complete.
    for (let i = 1; i < PIPELINE_STAGES.length; i++) {
      stepTimers.current.push(setTimeout(() => setActiveStage(i), i * 650));
    }
  };

  const executePipeline = async (payload, meta) => {
    setRunning(true);
    setRunComplete(false);
    setRunError(null);
    setResult(null);
    startStepperAnimation();
    try {
      const res = await runPipeline(payload);
      // The backend now provides a base order_details (including inventory).
      // We merge the Command Center inputs over it so the topbar pill + downstream tabs stay consistent.
      const enriched = {
        ...res.data,
        order_details: {
          ...(res.data.order_details || {}),
          city: meta.hub,
          value: meta.orderValue ?? meta.quantity * 100,
          quantity: meta.quantity,
          shipping_mode: meta.urgency,
          department: "Command Center",
        },
      };
      setResult(enriched);
      setLiveOrder?.(enriched); // persist to shared App state → RL Agent tab reads it
      // Ensure the stepper visibly reaches the final stage before completing.
      setActiveStage(PIPELINE_STAGES.length - 1);
      setRunComplete(true);
    } catch (e) {
      setRunError(
        e?.response?.data?.detail ||
          "Pipeline request failed. Is the backend running and are all models trained?"
      );
    } finally {
      stepTimers.current.forEach(clearTimeout);
      stepTimers.current = [];
      setRunning(false);
    }
  };

  const handleQuickSim = () => {
    setShowForm(false);
    const meta = { quantity: 50, urgency: "Standard", hub: "Bengaluru" };
    executePipeline(buildPayload(meta), meta);
  };

  const handleCustomExecute = () => {
    executePipeline(buildPayload(form), {
      quantity: Number(form.quantity) || 1,
      urgency: form.urgency,
      hub: form.hub,
      orderValue: Number(form.orderValue) || undefined,
    });
  };

  useEffect(() => () => stepTimers.current.forEach(clearTimeout), []);

  useEffect(() => {
    const poll = async () => {
      try {
        const r = await getStatus();
        setStatus(r.data);
      } catch {
        setStatus(null);
      }
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  const health = [
    { label: "API", value: status ? "Running" : "Offline", ok: !!status, icon: CheckCircle2 },
    { label: "Dataset", value: status?.data_available ? "Loaded" : "Missing", ok: status?.data_available, icon: BarChart3 },
    { label: "Models", value: status?.models_loaded ? "Ready" : status?.training_in_progress ? "Training..." : "Pending", ok: status?.models_loaded, icon: Brain },
    { label: "Training", value: status?.training_in_progress ? "In progress" : status?.models_loaded ? "Complete" : "Queued", ok: status?.models_loaded || status?.training_in_progress, icon: Clock },
  ];

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
      <section className="dashboard-hero">
        <div className="hero-copy">
          <div className="hero-eyebrow">
            <Zap size={12} /> AI-powered supply chain intelligence
          </div>
          <h1 className="hero-title">
            SmartChain <span>AI</span>
          </h1>
          <p className="hero-text">
            A light, live operations cockpit for demand forecasting, delivery risk, adaptive routing, and inventory decisions.
          </p>
          <div className="hero-kpis">
            {HERO_KPIS.map((item) => (
              <div className="hero-kpi" key={item.label}>
                <div className="hero-kpi-value">{item.value}</div>
                <div className="hero-kpi-label">{item.label}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="hero-visual">
          <NetworkGraphic />
        </div>
      </section>

      {/* ─── Command Center: dual triggers ─────────────────────────────────── */}
      <div className="card section-gap">
        <p className="card-label">Command Center</p>
        <p className="card-title">Execute AI Pipeline</p>
        <p className="card-subtitle">
          Run a one-click standard simulation or configure a custom live order.
        </p>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 16 }}>
          <motion.button
            className="btn btn-primary btn-lg"
            onClick={handleQuickSim}
            disabled={running}
            whileTap={{ scale: 0.97 }}
          >
            {running ? <Loader2 size={16} className="spin" /> : <Rocket size={16} />}
            Quick Simulation
          </motion.button>
          <motion.button
            className="btn btn-secondary btn-lg"
            onClick={() => setShowForm((s) => !s)}
            disabled={running}
            whileTap={{ scale: 0.97 }}
          >
            {showForm ? <X size={16} /> : <SlidersHorizontal size={16} />}
            Custom Live Order
          </motion.button>
        </div>

        {/* Inline custom-order form */}
        <AnimatePresence>
          {showForm && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              style={{ overflow: "hidden" }}
            >
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                  gap: 16,
                  marginTop: 20,
                  padding: 20,
                  background: "var(--bg-card-hover)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--r-md)",
                }}
              >
                <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="card-label">Target Quantity</span>
                  <input
                    type="number"
                    min={1}
                    value={form.quantity}
                    onChange={(e) => setForm((f) => ({ ...f, quantity: e.target.value }))}
                    style={fieldStyle}
                  />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="card-label">Delivery Urgency</span>
                  <select
                    value={form.urgency}
                    onChange={(e) => setForm((f) => ({ ...f, urgency: e.target.value }))}
                    style={fieldStyle}
                  >
                    {Object.keys(URGENCY_PRESETS).map((u) => (
                      <option key={u} value={u}>{u}</option>
                    ))}
                  </select>
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="card-label">Destination Hub</span>
                  <select
                    value={form.hub}
                    onChange={(e) => setForm((f) => ({ ...f, hub: e.target.value }))}
                    style={fieldStyle}
                  >
                    {Object.keys(HUB_NODES).map((h) => (
                      <option key={h} value={h}>{h}</option>
                    ))}
                  </select>
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="card-label">Current Inventory</span>
                  <input
                    type="number"
                    min={0}
                    value={form.inventory}
                    onChange={(e) => setForm((f) => ({ ...f, inventory: e.target.value }))}
                    style={fieldStyle}
                  />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="card-label">Supplier Reliability</span>
                  <select
                    value={form.reliability}
                    onChange={(e) => setForm((f) => ({ ...f, reliability: e.target.value }))}
                    style={fieldStyle}
                  >
                    {Object.keys(RELIABILITY_PRESETS).map((r) => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="card-label">Source Warehouse</span>
                  <select
                    value={form.source}
                    onChange={(e) => setForm((f) => ({ ...f, source: e.target.value }))}
                    style={fieldStyle}
                  >
                    {Object.keys(SOURCE_NODES).map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="card-label">Order Value ($)</span>
                  <input
                    type="number"
                    min={1}
                    value={form.orderValue}
                    onChange={(e) => setForm((f) => ({ ...f, orderValue: e.target.value }))}
                    style={fieldStyle}
                  />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <span className="card-label">Discount Rate (0–1)</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={form.discountRate}
                    onChange={(e) => setForm((f) => ({ ...f, discountRate: e.target.value }))}
                    style={fieldStyle}
                  />
                </label>
                <div style={{ display: "flex", alignItems: "flex-end" }}>
                  <motion.button
                    className="btn btn-cyan"
                    onClick={handleCustomExecute}
                    disabled={running}
                    whileTap={{ scale: 0.97 }}
                    style={{ width: "100%" }}
                  >
                    {running ? <Loader2 size={16} className="spin" /> : <Zap size={16} />}
                    Execute
                  </motion.button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Sequential pipeline stepper — shown once a run starts */}
        <AnimatePresence>
          {(running || runComplete) && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <PipelineStepper activeIndex={activeStage} done={runComplete} />
            </motion.div>
          )}
        </AnimatePresence>

        {runError && (
          <div className="alert alert-danger" style={{ marginTop: 16 }}>
            <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
            <div>{runError}</div>
          </div>
        )}

        {/* Run complete → send the user to the diagnostic tab (RL Agent) */}
        <AnimatePresence>
          {runComplete && !running && !runError && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={{
                marginTop: 18,
                padding: "16px 18px",
                borderRadius: "var(--r-md)",
                border: "1px solid rgba(16,185,129,0.28)",
                background: "linear-gradient(135deg, rgba(16,185,129,0.08), rgba(37,99,235,0.05))",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: 12,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <CheckCircle2 size={20} style={{ color: "var(--emerald-500)" }} />
                <div>
                  <p style={{ fontWeight: 800, color: "var(--text-primary)", fontSize: 14 }}>
                    Run Complete — {result?.rl_action?.action?.replace(/_/g, " ") || "decision ready"}
                  </p>
                  <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    Full diagnostics, reward breakdown & AI chat are in the RL Agent tab.
                  </p>
                </div>
              </div>
              <motion.button
                className="btn btn-primary"
                onClick={() => onNavigate("rl")}
                whileTap={{ scale: 0.97 }}
              >
                View Details in RL Agent <ArrowRight size={15} />
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="card section-gap">
        <div className="health-grid">
          {health.map(({ label, value, ok, icon: Icon }) => (
            <div className="health-item" key={label}>
              <div className="health-icon" style={{ color: ok ? "var(--emerald-500)" : "var(--text-muted)" }}>
                <Icon size={16} />
              </div>
              <div>
                <p className="card-label" style={{ marginBottom: 3 }}>{label}</p>
                <p style={{ fontSize: 13, fontWeight: 800, color: ok ? "var(--text-primary)" : "var(--rose-500)" }}>{value}</p>
              </div>
            </div>
          ))}
        </div>

        <AnimatePresence>
          {status?.training_in_progress && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="glow-pill"
              style={{ marginTop: 14 }}
            >
              <div className="spinner" style={{ width: 12, height: 12, borderWidth: "1.5px" }} />
              Auto-training all models...
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {!status && (
        <div className="alert alert-danger section-gap">
          <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
          <div>
            <strong>Backend offline.</strong> Start the FastAPI server from the project root:<br />
            <code style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>python backend/main.py</code>
          </div>
        </div>
      )}

      <div className="card section-gap">
        <p className="card-label">Architecture</p>
        <p className="card-title">Integration Pipeline</p>
        <p className="card-subtitle">Raw order data is transformed into prediction, detection, routing, and inventory actions.</p>
        <div className="pipeline-row">
          {PIPELINE.map((step, i) => {
            const Icon = step.icon;
            return (
              <motion.div
                className="pipeline-node"
                key={step.label}
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <div className="pipeline-node-icon" style={{ background: step.color }}>
                  <Icon size={16} />
                </div>
                <div>
                  <p className="pipeline-node-title">{step.label}</p>
                  <p className="pipeline-node-sub">{step.sub}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      <div className="card">
        <p className="card-label">Navigation</p>
        <p className="card-title">Explore Modules</p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 16 }}>
          {[
            { id: "demand",  label: "Demand Forecast", cls: "btn-cyan",    icon: TrendingUp },
            { id: "anomaly", label: "Delivery Risk",   cls: "btn-primary", icon: AlertCircle },
            { id: "rl",      label: "RL Agent",        cls: "btn-danger",  icon: Package },
            { id: "history", label: "History",          cls: "btn-primary", icon: Activity },
            { id: "routing", label: "Route Optimizer",  cls: "btn-primary", icon: Network },
          ].map(({ id, label, cls, icon: Icon }) => (
            <button key={id} className={`btn ${cls}`} onClick={() => onNavigate(id)}>
              <Icon size={15} /> {label}
            </button>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
