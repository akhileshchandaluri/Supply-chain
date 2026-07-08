import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";

/* ── Risk classification config ────────────────────────────────────────────── */
const RISK_CFG = {
  LOW:    { color: "#10B981", bg: "rgba(16,185,129,0.08)",  border: "rgba(16,185,129,0.3)",  glow: "0 16px 40px rgba(16,185,129,0.12)" },
  MEDIUM: { color: "#F59E0B", bg: "rgba(245,158,11,0.08)",  border: "rgba(245,158,11,0.3)",  glow: "0 16px 40px rgba(245,158,11,0.12)" },
  HIGH:   { color: "#F43F5E", bg: "rgba(244,63,94,0.08)",   border: "rgba(244,63,94,0.3)",   glow: "0 16px 40px rgba(244,63,94,0.12)" },
};

/* ── Risk Classification Section (Random Forest) ───────────────────────────── */
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

            {/* Feature values from live order */}
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
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
      <div className="page-header">
        <div className="page-badge"><ShieldAlert size={10} /> Random Forest</div>
        <h1 className="page-title">Delivery Risk</h1>
        <p className="page-subtitle">
          Delivery risk classification (Random Forest) for the live order stream.
        </p>
      </div>

      <RiskSection liveOrder={liveOrder} />
    </motion.div>
  );
}
