import { motion } from "framer-motion";

export default function MetricCard({
  icon,
  label,
  value,
  change,
  changeDir,
  gradient,
  delay = 0,
}) {
  return (
    <motion.div
      className="metric-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
      style={{ "--gradient": gradient }}
    >
      {icon && (
        <div
          className="metric-icon"
          style={{
            background: gradient || "linear-gradient(135deg, #6366F1, #8B5CF6)",
          }}
        >
          {icon}
        </div>
      )}
      <p className="metric-label">{label}</p>
      <p className="metric-value">{value}</p>
      {change && (
        <p className={`metric-change ${changeDir}`}>
          {changeDir === "up" ? "↑" : "↓"} {change}
        </p>
      )}
    </motion.div>
  );
}
