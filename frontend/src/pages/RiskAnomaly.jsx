import { useCallback, useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend as RLegend,
} from "recharts";
import { detectAnomalies } from "../api/client";
import {
  AlertTriangle, CheckCircle2, RefreshCw, ShieldAlert, Search,
} from "lucide-react";

/* ── Risk classification config ────────────────────────────────────────────── */
const RISK_CFG = {
  LOW:    { color: "#10B981", bg: "rgba(16,185,129,0.08)",  border: "rgba(16,185,129,0.3)",  glow: "0 16px 40px rgba(16,185,129,0.12)" },
  MEDIUM: { color: "#F59E0B", bg: "rgba(245,158,11,0.08)",  border: "rgba(245,158,11,0.3)",  glow: "0 16px 40px rgba(245,158,11,0.12)" },
  HIGH:   { color: "#F43F5E", bg: "rgba(244,63,94,0.08)",   border: "rgba(244,63,94,0.3)",   glow: "0 16px 40px rgba(244,63,94,0.12)" },
};

/* ── Isolation Forest algorithm steps ──────────────────────────────────────── */
const ISO_STEPS = [
  { n: 1, title: "Build Isolation Trees",     desc: "100 trees randomly select a feature and split on a random threshold." },
  { n: 2, title: "Measure Path Length",        desc: "Anomalies are isolated in fewer splits — shorter average path length." },
  { n: 3, title: "Compute Anomaly Score",      desc: "Score near -1 → highly anomalous. Score near +1 → normal." },
  { n: 4, title: "Flag Anomalies",             desc: "Top 5% most isolated suppliers flagged (contamination=0.05)." },
  { n: 5, title: "Trigger Action",             desc: "Anomaly → RL agent selects SWITCH_SUPPLIER or EMERGENCY_REORDER." },
];

/* ── Custom Scatter Tooltip ────────────────────────────────────────────────── */
function ScatterTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div style={{
      background: "#fff", border: "1px solid var(--border)", borderRadius: 10,
      padding: "10px 14px", boxShadow: "0 8px 24px rgba(0,0,0,0.08)", fontSize: 12,
    }}>
      <p style={{ fontWeight: 800, marginBottom: 6, color: "var(--text-primary)" }}>{d.department}</p>
      <p>Avg Delivery: <strong>{d.avg_delivery_time}d</strong></p>
      <p>Price Dev: <strong>{d.price_deviation?.toFixed(4)}</strong></p>
      <p>Score: <strong style={{ color: d.is_anomaly ? "#E11D48" : "#10B981" }}>{d.anomaly_score}</strong></p>
    </div>
  );
}

/* ── Risk Classification Section ───────────────────────────────────────────── */
function RiskSection({ liveOrder }) {
  const result = liveOrder ? {
    label: liveOrder.risk_level,
    probabilities: liveOrder.risk_probabilities ? [
      liveOrder.risk_probabilities.LOW,
      liveOrder.risk_probabilities.MEDIUM,
      liveOrder.risk_probabilities.HIGH,
    ] : [0, 0, 0],
  } : null;

  const cfg = result ? RISK_CFG[result.label] : null;
  const RiskIcon = result?.label === "LOW" ? CheckCircle2 : AlertTriangle;

  return (
    <div className="card section-gap">
      <p className="card-label">Risk Engine</p>
      <p className="card-title">Delivery Risk Classification</p>
      <p className="card-subtitle">Random Forest 3-class prediction based on the live order's shipping features.</p>

      <AnimatePresence mode="wait">
        {result && cfg ? (
          <motion.div
            key={result.label}
            initial={{ opacity: 0, scale: 0.94 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
          >
            <div style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 20,
              marginTop: 16,
            }}>
              {/* Risk badge */}
              <div style={{
                textAlign: "center", padding: "32px 20px",
                background: cfg.bg, border: `1px solid ${cfg.border}`,
                borderRadius: "var(--r-lg)", boxShadow: cfg.glow,
              }}>
                <div style={{
                  width: 56, height: 56, borderRadius: "var(--r-lg)", margin: "0 auto",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  background: "#FFFFFF", color: cfg.color, border: `1px solid ${cfg.border}`,
                }}>
                  <RiskIcon size={28} />
                </div>
                <p style={{
                  fontFamily: "var(--font-display)", fontSize: 30, fontWeight: 900,
                  color: cfg.color, marginTop: 12,
                }}>{result.label}</p>
                <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>Delivery Risk Level</p>
              </div>

              {/* Probability bars */}
              <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", gap: 14 }}>
                {["LOW", "MEDIUM", "HIGH"].map((cls, i) => {
                  const pct = (result.probabilities[i] * 100).toFixed(1);
                  const c = RISK_CFG[cls];
                  return (
                    <div key={cls}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: c.color }}>{cls}</span>
                        <span style={{ fontSize: 12, fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--text-primary)" }}>{pct}%</span>
                      </div>
                      <div className="score-bar-track">
                        <motion.div className="score-bar-fill"
                          style={{ background: c.color, width: 0 }}
                          animate={{ width: `${pct}%` }}
                          transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Feature importances from live order */}
            {liveOrder?.model_inputs?.risk_features && (
              <div style={{ marginTop: 20 }}>
                <p style={{ fontSize: 10, fontWeight: 800, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
                  Order Risk Features
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {Object.entries(liveOrder.model_inputs.risk_features).map(([k, v]) => (
                    <div key={k} style={{
                      padding: "5px 10px", borderRadius: 6,
                      border: "1px solid var(--border)", background: "#f8fafc",
                      fontSize: 11, fontFamily: "var(--font-mono)",
                    }}>
                      <span style={{ color: "var(--text-muted)" }}>{k.replace(/_/g, " ")}:</span>{" "}
                      <strong>{typeof v === "number" ? v.toFixed(2) : v}</strong>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div key="empty" className="loading-center"
            style={{ border: "1px dashed var(--border)", borderRadius: "var(--r-lg)", marginTop: 16, minHeight: 180 }}>
            <ShieldAlert size={28} style={{ color: "var(--text-dim)" }} />
            <span>Run a live order to see risk classification</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ── Main page component ───────────────────────────────────────────────────── */
export default function RiskAnomaly({ liveOrder }) {
  const [anomalyData, setAnomalyData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadAnomalies = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await detectAnomalies();
      setAnomalyData(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || "Could not load anomaly data. Models may still be training.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      try {
        const res = await detectAnomalies();
        if (!cancelled) { setAnomalyData(res.data); setError(""); }
      } catch (e) {
        if (!cancelled) setError(e.response?.data?.detail || "Models not ready yet.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    init();
    return () => { cancelled = true; };
  }, []);

  // Re-fetch anomalies when a new live order comes in (supplier context may change)
  useEffect(() => {
    if (!liveOrder?.order_details?.id) return undefined;

    let cancelled = false;
    const refresh = async () => {
      try {
        const res = await detectAnomalies();
        if (!cancelled) {
          setAnomalyData(res.data);
          setError("");
        }
      } catch (e) {
        if (!cancelled) {
          setError(e.response?.data?.detail || "Could not refresh anomaly data.");
        }
      }
    };

    refresh();
    return () => { cancelled = true; };
  }, [liveOrder?.order_details?.id]);

  const suppliers = useMemo(() => anomalyData?.suppliers || [], [anomalyData?.suppliers]);
  const totalSuppliers = anomalyData?.total || suppliers.length;
  const anomalyCount = anomalyData?.anomaly_count || 0;
  const anomalyRate = anomalyData?.anomaly_rate || 0;

  // Scatter data: avg_delivery_time vs price_deviation
  const scatterData = useMemo(() =>
    suppliers.map((s) => ({
      ...s,
      x: s.avg_delivery_time,
      y: s.price_deviation,
    })),
    [suppliers]
  );

  // Compute axis domains with padding
  const xDomain = useMemo(() => {
    if (!scatterData.length) return [0, 5];
    const vals = scatterData.map(d => d.x);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const pad = (max - min) * 0.15 || 0.5;
    return [Math.max(0, min - pad), max + pad];
  }, [scatterData]);

  const yDomain = useMemo(() => {
    if (!scatterData.length) return [0, 1];
    const vals = scatterData.map(d => d.y);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const pad = (max - min) * 0.15 || 0.1;
    return [Math.max(0, min - pad), max + pad];
  }, [scatterData]);

  // Supplier context from live order
  const supplierCtx = liveOrder?.supplier_context;

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
      <div className="page-header">
        <div className="page-badge"><ShieldAlert size={10} /> Random Forest + Isolation Forest</div>
        <h1 className="page-title">Risk & Anomaly Detection</h1>
        <p className="page-subtitle">
          Delivery risk classification (Random Forest) and supplier anomaly detection (Isolation Forest) for the live order stream.
        </p>
      </div>

      {/* ── Section 1: Risk Classification ─────────────────────────────────── */}
      <RiskSection liveOrder={liveOrder} />

      {/* ── Supplier context from live order ───────────────────────────────── */}
      {supplierCtx && (
        <motion.div
          className="card section-gap"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            borderLeft: `4px solid ${supplierCtx.is_anomaly ? "#F43F5E" : "#10B981"}`,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
            <div>
              <p className="card-label">Current Order Supplier</p>
              <p className="card-title" style={{ fontSize: 18 }}>{supplierCtx.department}</p>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <span className={`badge ${supplierCtx.is_anomaly ? "badge-anomaly" : "badge-normal"}`}>
                {supplierCtx.is_anomaly ? "⚠ Anomaly" : "✓ Normal"}
              </span>
              <span className="badge badge-violet" style={{ fontFamily: "var(--font-mono)" }}>
                Score: {supplierCtx.score?.toFixed(4)}
              </span>
            </div>
          </div>
          <div style={{ display: "flex", gap: 16, marginTop: 12, fontSize: 12, color: "var(--text-secondary)" }}>
            <span>Delivery: <strong>{supplierCtx.metrics?.avg_delivery_time?.toFixed(2)}d</strong></span>
            <span>Fulfillment: <strong>{(supplierCtx.metrics?.fulfillment_rate * 100)?.toFixed(1)}%</strong></span>
            <span>Delay Rate: <strong>{(supplierCtx.delay_rate * 100)?.toFixed(1)}%</strong></span>
          </div>
        </motion.div>
      )}

      {/* ── Section 2: Anomaly Detection ───────────────────────────────────── */}
      <div className="page-header" style={{ marginTop: 8 }}>
        <div className="page-badge"><Search size={10} /> Isolation Forest</div>
        <h1 className="page-title" style={{ fontSize: "clamp(22px, 2.5vw, 32px)" }}>Supplier Anomaly Detection</h1>
        <p className="page-subtitle">
          Unsupervised supplier anomaly detection using Isolation Forest (contamination=5%).
        </p>
      </div>

      {/* Summary cards */}
      {!loading && !error && anomalyData && (
        <div className="metrics-grid section-gap">
          {[
            { l: "Total Suppliers", v: totalSuppliers, g: "linear-gradient(135deg,#2563eb,#38bdf8)" },
            { l: "Anomalies", v: anomalyCount, g: "linear-gradient(135deg,#F43F5E,#be123c)" },
            { l: "Anomaly Rate", v: `${anomalyRate}%`, g: "linear-gradient(135deg,#F59E0B,#f97316)" },
            { l: "Normal", v: totalSuppliers - anomalyCount, g: "linear-gradient(135deg,#10b981,#0d9488)" },
          ].map(({ l, v, g }, i) => (
            <motion.div key={l} className="metric-card"
              initial={{ opacity: 0, y: 20, scale: 0.92 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ type: "spring", stiffness: 300, damping: 20, delay: i * 0.05 }}
              style={{ "--gradient": g }}
            >
              <div className="metric-icon" style={{ background: g }}>
                {l === "Anomalies" ? <AlertTriangle size={17} /> :
                  l === "Normal" ? <CheckCircle2 size={17} /> :
                    <Search size={17} />}
              </div>
              <p className="metric-label">{l}</p>
              <p className="metric-value">{v}</p>
            </motion.div>
          ))}
        </div>
      )}

      {/* Scatter + Algorithm explanation */}
      {!loading && !error && anomalyData && (
        <div className="grid-2 section-gap">
          {/* Scatter chart */}
          <div className="card" style={{ padding: 22 }}>
            <p className="card-label">Visual Map</p>
            <p className="card-title">Supplier Anomaly Scatter</p>
            <p className="card-subtitle">Delivery time vs. price deviation — red = anomaly</p>

            <ResponsiveContainer width="100%" height={340}>
              <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  type="number" dataKey="x" name="Avg Delivery"
                  domain={xDomain}
                  tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                  axisLine={false} tickLine={false}
                  label={{ value: "Avg Delivery (d)", position: "bottom", offset: 0, fontSize: 11, fill: "var(--text-muted)" }}
                />
                <YAxis
                  type="number" dataKey="y" name="Price Deviation"
                  domain={yDomain}
                  tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                  axisLine={false} tickLine={false}
                  label={{ value: "Price Deviation", angle: -90, position: "insideLeft", offset: -2, fontSize: 11, fill: "var(--text-muted)" }}
                />
                <Tooltip content={<ScatterTooltip />} />
                <RLegend
                  wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
                  formatter={(val) => val === "normal" ? "Normal" : "Anomaly"}
                />
                <Scatter name="normal" data={scatterData.filter(d => !d.is_anomaly)} fill="#6366F1">
                  {scatterData.filter(d => !d.is_anomaly).map((_, i) => (
                    <Cell key={i} fill="#6366F1" r={5} />
                  ))}
                </Scatter>
                <Scatter name="anomaly" data={scatterData.filter(d => d.is_anomaly)} fill="#E11D48">
                  {scatterData.filter(d => d.is_anomaly).map((_, i) => (
                    <Cell key={i} fill="#E11D48" r={7} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Algorithm steps */}
          <div className="card" style={{ padding: 22 }}>
            <p className="card-label">Algorithm</p>
            <p className="card-title">How Isolation Forest Works</p>
            <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 14 }}>
              {ISO_STEPS.map((step) => (
                <div key={step.n} style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                  <div style={{
                    width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    background: "linear-gradient(135deg,#10b981,#0d9488)",
                    color: "#fff", fontSize: 13, fontWeight: 800,
                  }}>{step.n}</div>
                  <div>
                    <p style={{ fontWeight: 800, fontSize: 13, color: "#10B981", marginBottom: 3 }}>{step.title}</p>
                    <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.55 }}>{step.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Supplier Scorecard Table */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div>
            <p className="card-label">Supplier Scorecard</p>
            <p className="card-title">All Departments (sorted by anomaly score ↑)</p>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={loadAnomalies} disabled={loading}>
            <RefreshCw size={12} className={loading ? "spin" : ""} /> Refresh
          </button>
        </div>

        {loading && <div className="loading-center"><div className="spinner spinner-lg" /><span>Loading supplier data...</span></div>}
        {error && <div className="alert alert-warning">{error}</div>}

        {!loading && !error && suppliers.length > 0 && (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Department</th>
                  <th>Avg Delivery</th>
                  <th>Fulfillment</th>
                  <th>Complaints</th>
                  <th>Score</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {[...suppliers]
                  .sort((a, b) => a.anomaly_score - b.anomaly_score)
                  .map((s) => (
                    <tr key={s.department} style={{
                      background: s.is_anomaly ? "rgba(244,63,94,0.03)" : undefined,
                    }}>
                      <td style={{ fontWeight: 800 }}>{s.department}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{s.avg_delivery_time?.toFixed(2)}d</td>
                      <td>{(s.fulfillment_rate * 100)?.toFixed(1)}%</td>
                      <td>{s.complaint_freq}</td>
                      <td style={{
                        fontFamily: "var(--font-mono)", fontWeight: 700,
                        color: s.is_anomaly ? "#E11D48" : "#10B981",
                      }}>{s.anomaly_score?.toFixed(4)}</td>
                      <td>
                        <span className={`badge ${s.is_anomaly ? "badge-anomaly" : "badge-normal"}`}>
                          {s.is_anomaly ? "ANOMALY" : "Normal"}
                        </span>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </motion.div>
  );
}
