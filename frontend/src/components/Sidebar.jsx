import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  LayoutDashboard, TrendingUp, AlertTriangle,
  Network, BrainCircuit, Zap, History,
} from "lucide-react";
import { getStatus } from "../api/client";

const NAV = [
  { id: "dashboard", label: "Overview",          icon: LayoutDashboard, badge: null     },
  { id: "demand",    label: "Demand Forecast",   icon: TrendingUp,      badge: "XGBOOST"},
  { id: "anomaly",   label: "Delivery Risk",     icon: AlertTriangle,   badge: "RF"     },
  { id: "rl",        label: "RL Agent",          icon: BrainCircuit,    badge: "QL"     },
  { id: "history",   label: "History",           icon: History,         badge: "LOG"    },
  { id: "routing",   label: "Route Optimizer",   icon: Network,         badge: "FINAL"  },
];

export default function Sidebar({ activePage, onNavigate }) {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    const poll = async () => {
      try { const r = await getStatus(); setStatus(r.data); }
      catch { setStatus(null); }
    };
    poll();
    const id = setInterval(poll, 4000);
    return () => clearInterval(id);
  }, []);

  const dotClass = !status ? "offline"
    : status.training_in_progress ? "training"
    : "online";

  const statusText = !status ? "Offline"
    : status.training_in_progress ? "Training models..."
    : status.models_loaded ? "Ready"
    : "Awaiting training";

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-mark"><Zap size={20} /></div>
        <div>
          <div className="logo-text">SmartChain</div>
          <div className="logo-sub">AI Supply Chain</div>
        </div>
      </div>

      <p className="sidebar-section">Modules</p>
      <nav className="sidebar-nav">
        {NAV.map(({ id, label, icon: Icon, badge }) => (
          <motion.button
            key={id}
            type="button"
            className={`nav-item ${activePage === id ? "active" : ""}`}
            onClick={() => onNavigate(id)}
            aria-current={activePage === id ? "page" : undefined}
            whileTap={{ scale: 0.97 }}
          >
            {activePage === id && <motion.span className="nav-active-pill" layoutId="nav-active-pill" />}
            <span className="nav-icon"><Icon size={15} /></span>
            <span className="nav-label">{label}</span>
            {badge && <span className="nav-badge">{badge}</span>}
          </motion.button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-indicator">
          <div className={`status-dot ${dotClass}`} />
          <span>{statusText}</span>
        </div>
        {status?.training_in_progress && (
          <motion.div
            style={{ fontSize: 10, color: "var(--violet-400)", marginTop: 6,
              display:"flex", alignItems:"center", gap:5 }}
            animate={{ opacity: [0.5,1,0.5] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >
            <Zap size={10} /> Auto-training in background...
          </motion.div>
        )}
      </div>
    </aside>
  );
}
