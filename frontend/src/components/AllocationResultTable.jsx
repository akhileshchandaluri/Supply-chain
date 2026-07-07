import { motion } from "framer-motion";
import { PackageCheck } from "lucide-react";

// Map solver status → existing dashboard badge classes.
const STATUS_BADGE = {
  OPTIMAL: "badge-normal", // green
  FEASIBLE: "badge-medium", // amber
  INFEASIBLE: "badge-high", // red
  ABNORMAL: "badge-high",
  SOLVER_UNAVAILABLE: "badge-high",
};

const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const titleCase = (s) =>
  typeof s === "string" ? s.charAt(0).toUpperCase() + s.slice(1) : s;

export default function AllocationResultTable({ optimization }) {
  // Nothing to show until the pipeline has produced an optimization block.
  if (!optimization) return null;

  const { status, total_optimized_cost, allocations = [] } = optimization;
  const badgeClass = STATUS_BADGE[status] || "badge-violet";

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      style={{ padding: 24 }}
    >
      {/* Header: label/title on the left, status badge on the right */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: 16,
        }}
      >
        <div>
          <p className="card-label">Optimization Layer 4.5</p>
          <p className="card-title" style={{ fontSize: 15, marginBottom: 2 }}>
            Warehouse Allocation Plan
          </p>
        </div>
        <span className={`badge ${badgeClass}`}>
          <PackageCheck size={11} /> {status}
        </span>
      </div>

      {allocations.length > 0 ? (
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>From Warehouse</th>
                <th>To Region</th>
                <th style={{ textAlign: "right" }}>Units</th>
              </tr>
            </thead>
            <tbody>
              {allocations.map((a, i) => (
                <tr key={`${a.warehouse}-${a.region}-${i}`}>
                  <td>{titleCase(a.warehouse)}</td>
                  <td>{titleCase(a.region)}</td>
                  <td
                    style={{
                      textAlign: "right",
                      fontFamily: "var(--font-mono)",
                      color: "var(--text-primary)",
                      fontWeight: 700,
                    }}
                  >
                    {Number(a.units).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr
                style={{
                  borderTop: "1px solid var(--border-strong)",
                  background: "#f8fbff",
                }}
              >
                <td
                  colSpan={2}
                  style={{
                    padding: "12px 16px",
                    fontWeight: 800,
                    fontSize: 11,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    color: "var(--text-muted)",
                  }}
                >
                  Total Optimized Cost
                </td>
                <td
                  style={{
                    padding: "12px 16px",
                    textAlign: "right",
                    fontFamily: "var(--font-mono)",
                    fontWeight: 800,
                    fontSize: 14,
                    color: "var(--blue-600)",
                  }}
                >
                  {typeof total_optimized_cost === "number"
                    ? currency.format(total_optimized_cost)
                    : "—"}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      ) : (
        // Solver ran but produced no shipments (e.g. INFEASIBLE / zero demand).
        <div className="alert alert-warning">
          No allocations were produced for this plan
          {status ? ` (status: ${status})` : ""}.
        </div>
      )}
    </motion.div>
  );
}
