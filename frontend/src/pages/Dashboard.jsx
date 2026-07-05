import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { getStatus } from "../api/client";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Brain,
  CheckCircle2,
  Clock,
  Database,
  History,
  Network,
  Package,
  ScanSearch,
  Settings,
  TrendingUp,
  Zap,
} from "lucide-react";

const PIPELINE = [
  { icon: Database,    label: "DataCo CSV",    sub: "Order history",    color: "linear-gradient(135deg,#2563eb,#38bdf8)" },
  { icon: Settings,    label: "Feature Eng.",   sub: "Lag and rolling sets", color: "linear-gradient(135deg,#0d9488,#2dd4bf)" },
  { icon: TrendingUp,  label: "XGBoost",        sub: "Demand forecast",  color: "linear-gradient(135deg,#06b6d4,#2563eb)" },
  { icon: BarChart3,   label: "Random Forest",  sub: "Risk classes",     color: "linear-gradient(135deg,#10b981,#0d9488)" },
  { icon: ScanSearch,  label: "Isolation F.",    sub: "Supplier signals", color: "linear-gradient(135deg,#f59e0b,#f97316)" },
  { icon: Brain,       label: "Q-Learning",     sub: "Inventory action", color: "linear-gradient(135deg,#f43f5e,#8b5cf6)" },
  { icon: History,     label: "History",         sub: "Decision audit",   color: "linear-gradient(135deg,#64748b,#2563eb)" },
  { icon: Network,     label: "Routing",         sub: "A* and Dijkstra",  color: "linear-gradient(135deg,#7c3aed,#2563eb)" },
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

export default function Dashboard({ onNavigate, liveOrder }) {
  const [status, setStatus] = useState(null);

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
            A light, live operations cockpit for demand forecasting, delivery risk, supplier anomalies, adaptive routing, and inventory decisions.
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

      <AnimatePresence>
        {liveOrder && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="card section-gap"
            style={{ borderLeft: "4px solid var(--blue-500)" }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16 }}>
              <div>
                <p className="card-label">Active Order</p>
                <p className="card-title" style={{ fontSize: 24, marginBottom: 12 }}>{liveOrder.order_details?.city} Delivery</p>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <div className="badge badge-violet">{liveOrder.order_details?.shipping_mode}</div>
                  <div className="badge badge-normal">${liveOrder.order_details?.value?.toFixed(2)}</div>
                  <div className="badge badge-medium">Dept: {liveOrder.order_details?.department}</div>
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <p className="card-label" style={{ marginBottom: 8 }}>Pipeline Execution</p>
                <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
                  <div className="badge badge-normal">{liveOrder.pipeline_summary?.step_1_demand}</div>
                  <div className={`badge ${liveOrder.risk_level === "HIGH" ? "badge-high" : liveOrder.risk_level === "MEDIUM" ? "badge-medium" : "badge-low"}`}>{liveOrder.pipeline_summary?.step_2_risk}</div>
                  <div className={`badge ${liveOrder.anomaly?.is_anomaly ? "badge-anomaly" : "badge-normal"}`}>{liveOrder.pipeline_summary?.step_3_anomaly}</div>
                  <div className="badge badge-violet">{liveOrder.pipeline_summary?.step_4_rl}</div>
                  <div className={`badge ${liveOrder.is_emergency ? "badge-high" : "badge-normal"}`}>{liveOrder.pipeline_summary?.step_5_route}</div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

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
            { id: "anomaly", label: "Risk & Anomaly",  cls: "btn-primary", icon: AlertCircle },
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
