import { motion } from "framer-motion";
import { History as HistoryIcon, Route, ShieldAlert, Trash2 } from "lucide-react";

const ACTION_COLORS = {
  HOLD: "#6366F1",
  REORDER_SMALL: "#10B981",
  REORDER_MEDIUM: "#F59E0B",
  REORDER_LARGE: "#FB923C",
  EMERGENCY_REORDER: "#F43F5E",
  SWITCH_SUPPLIER: "#A78BFA",
};

function formatTime(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return "—";
  }
}

export default function History({ history = [], onNavigate, setHistory }) {
  const handleClear = () => {
    if (setHistory) setHistory([]);
  };

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
      <div className="page-header">
        <div className="page-badge"><HistoryIcon size={10} /> Order audit trail</div>
        <h1 className="page-title">Live Order History</h1>
        <p className="page-subtitle">Every simulated order is recorded here with the model decisions and route selected by the pipeline.</p>
      </div>

      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center", marginBottom: 18, flexWrap: "wrap" }}>
          <div>
            <p className="card-label">Session History</p>
            <p className="card-title">{history.length} processed orders</p>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-secondary btn-sm" onClick={() => onNavigate?.("routing")} disabled={!history.length}>
              <Route size={13} /> View latest route
            </button>
            {history.length > 0 && (
              <button className="btn btn-secondary btn-sm" onClick={handleClear} style={{ color: "var(--rose-500)" }}>
                <Trash2 size={13} /> Clear
              </button>
            )}
          </div>
        </div>

        {!history.length ? (
          <div className="loading-center" style={{ border: "1px dashed var(--border)", borderRadius: 8 }}>
            <HistoryIcon size={32} style={{ color: "var(--text-dim)" }} />
            <span>Run a live order to start the audit trail.</span>
          </div>
        ) : (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Time</th>
                  <th>Order</th>
                  <th>Dept.</th>
                  <th>Value</th>
                  <th>Risk</th>
                  <th>RL Action</th>
                  <th>Route</th>
                </tr>
              </thead>
              <tbody>
                {history.map((row, i) => {
                  const actionColor = ACTION_COLORS[row.action] || "var(--text-secondary)";
                  return (
                    <tr key={`${row.order_id}-${i}`}>
                      <td>{history.length - i}</td>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)", whiteSpace: "nowrap" }}>
                        {formatTime(row.created_at)}
                      </td>
                      <td>
                        <div style={{ fontWeight: 800 }}>{row.city || "Unknown City"}</div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{row.order_id}</div>
                      </td>
                      <td>{row.department}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>${Number(row.value || 0).toFixed(2)}</td>
                      <td>
                        <span className={`badge ${row.risk === "HIGH" ? "badge-high" : row.risk === "MEDIUM" ? "badge-medium" : "badge-low"}`}>
                          <ShieldAlert size={11} /> {row.risk}
                        </span>
                      </td>
                      <td>
                        <span style={{
                          fontWeight: 700, fontSize: 12, color: actionColor,
                        }}>
                          {String(row.action || "").replaceAll("_", " ")}
                        </span>
                      </td>
                      <td style={{ minWidth: 260 }}>
                        <div style={{ fontWeight: 800, color: "var(--text-primary)" }}>{row.route_type}</div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{row.path}</div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </motion.div>
  );
}
