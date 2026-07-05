import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, Loader2, Play } from "lucide-react";
import { simulateOrder } from "./api/client";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import DemandForecast from "./pages/DemandForecast";
import RiskAnomaly from "./pages/RiskAnomaly";
import RLAgent from "./pages/RLAgent";
import History from "./pages/History";
import RouteOptimization from "./pages/RouteOptimization";
import "./index.css";

const PAGE_COMPONENTS = {
  dashboard: Dashboard,
  demand:    DemandForecast,
  anomaly:   RiskAnomaly,
  rl:        RLAgent,
  history:   History,
  routing:   RouteOptimization,
};

const PAGE_META = {
  dashboard: { kicker: "Command center", title: "Supply chain intelligence", subtitle: "Monitor demand, risk, anomalies, inventory decisions, history, and final routing." },
  demand:    { kicker: "Forecasting", title: "Demand forecast", subtitle: "Inspect actual demand history and the next prediction window." },
  anomaly:   { kicker: "Risk and supplier health", title: "Order risk & anomaly detection", subtitle: "Review the live order classification and supplier anomaly signals." },
  rl:        { kicker: "Inventory agent", title: "RL decision console", subtitle: "Inspect the action selected from the Q-table." },
  history:   { kicker: "Audit trail", title: "Order history", subtitle: "Compare previous simulated orders and decisions." },
  routing:   { kicker: "Final pipeline step", title: "Route optimizer", subtitle: "See the chosen network path for the latest order." },
};

const PAGE_TRANSITION = {
  initial:  { opacity: 0, y: 14, filter: "blur(8px)" },
  animate:  { opacity: 1, y: 0, filter: "blur(0px)" },
  exit:     { opacity: 0, y: -10, filter: "blur(8px)" },
  transition: { duration: 0.32, ease: [0.22, 1, 0.36, 1] },
};

export default function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [liveOrder, setLiveOrder] = useState(null);
  const [history, setHistory] = useState([]);
  const [loadingOrder, setLoadingOrder] = useState(false);

  const handleSimulate = async () => {
    setLoadingOrder(true);
    try {
      const res = await simulateOrder();
      setLiveOrder(res.data);
      if (res.data?.history_row) {
        setHistory((rows) => [
          { ...res.data.history_row, created_at: new Date().toISOString() },
          ...rows,
        ].slice(0, 50));
      }
    } catch (e) {
      console.error("Failed to simulate order", e);
    }
    setLoadingOrder(false);
  };

  const PageComponent = PAGE_COMPONENTS[activePage] || Dashboard;
  const activeMeta = PAGE_META[activePage] || PAGE_META.dashboard;

  return (
    <div className="layout">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <div className="app-body">
        <header className="topbar">
          <div className="topbar-copy">
            <p className="topbar-kicker">{activeMeta.kicker}</p>
            <h2 className="topbar-title">{activeMeta.title}</h2>
            <p className="topbar-subtitle">{activeMeta.subtitle}</p>
          </div>

          <div className="topbar-actions">
            <div className="live-status-pill">
              <Activity size={14} />
              {liveOrder?.order_details
                ? `${liveOrder.order_details.city || "Live"} order active`
                : "No live order yet"}
            </div>
            <motion.button
              className="btn btn-primary"
              onClick={handleSimulate}
              disabled={loadingOrder}
              whileTap={{ scale: 0.97 }}
            >
              {loadingOrder
                ? <Loader2 size={16} className="spin" />
                : <Play size={16} fill="currentColor" />}
              {loadingOrder ? "Processing..." : "Run Live Order"}
            </motion.button>
          </div>
        </header>
        <main className="main-content">
          <AnimatePresence mode="wait">
            <motion.div className="page-shell" key={activePage} {...PAGE_TRANSITION}>
              <PageComponent onNavigate={setActivePage} liveOrder={liveOrder} history={history} setHistory={setHistory} />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
