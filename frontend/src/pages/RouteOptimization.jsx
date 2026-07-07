import { Fragment, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Network } from "lucide-react";

// ─── Graph data mirrored from graph_construction.py ───────────────────────────
const LAT_MIN = 9.0, LAT_MAX = 32.0;
const LON_MIN = 71.0, LON_MAX = 93.0;
const SVG_W = 600, SVG_H = 480;
const PAD_X = 45, PAD_Y = 35;

const RAW_NODES = {
  0:  { name: "Mumbai",      lat: 19.0760, lon: 72.8777, type: "warehouse" },
  1:  { name: "Delhi",       lat: 28.6139, lon: 77.2090, type: "warehouse" },
  2:  { name: "Chennai",     lat: 13.0827, lon: 80.2707, type: "warehouse" },
  3:  { name: "Pune",        lat: 18.5204, lon: 73.8567, type: "supplier"  },
  4:  { name: "Ahmedabad",   lat: 23.0225, lon: 72.5714, type: "supplier"  },
  5:  { name: "Hyderabad",   lat: 17.3850, lon: 78.4867, type: "supplier"  },
  6:  { name: "Bengaluru",   lat: 12.9716, lon: 77.5946, type: "hub"       },
  7:  { name: "Kolkata",     lat: 22.5726, lon: 88.3639, type: "hub"       },
  8:  { name: "Jaipur",      lat: 26.9124, lon: 75.7873, type: "hub"       },
  9:  { name: "Surat",       lat: 21.1702, lon: 72.8311, type: "hub"       },
  10: { name: "Nagpur",      lat: 21.1458, lon: 79.0882, type: "supplier"  },
  11: { name: "Kochi",       lat:  9.9312, lon: 76.2673, type: "warehouse" },
  12: { name: "Indore",      lat: 22.7196, lon: 75.8577, type: "hub"       },
  13: { name: "Coimbatore",  lat: 11.0168, lon: 76.9558, type: "supplier"  },
  14: { name: "Bhopal",      lat: 23.2599, lon: 77.4126, type: "hub"       },
  15: { name: "Lucknow",     lat: 26.8467, lon: 80.9462, type: "supplier"  },
  16: { name: "Chandigarh",  lat: 30.7333, lon: 76.7794, type: "hub"       },
  17: { name: "Guwahati",    lat: 26.1445, lon: 91.7362, type: "warehouse" },
};

function project(lat, lon) {
  const x = PAD_X + ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * (SVG_W - 2 * PAD_X);
  const y = SVG_H - PAD_Y - ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * (SVG_H - 2 * PAD_Y);
  return { x: Math.round(x), y: Math.round(y) };
}

const NODE_POS = Object.fromEntries(
  Object.entries(RAW_NODES).map(([id, n]) => [
    Number(id),
    { ...project(n.lat, n.lon), ...n },
  ])
);

const EDGES = [
  [0,3,148],[0,6,984],[0,9,274],[1,8,268],[1,15,511],[1,16,248],
  [2,5,626],[2,13,508],[3,6,840],[3,9,126],[4,9,265],[4,8,668],
  [5,6,570],[5,10,699],[6,11,914],[6,13,358],[7,17,584],[8,12,483],
  [8,14,445],[9,12,448],[10,14,492],[11,13,215],[12,14,185],[14,15,552],
  [15,16,518],[16,17,871],[1,7,1495],[7,17,584],
];

const TYPE_STYLE = {
  warehouse: { fill: "#6366F1", glow: "rgba(99,102,246,0.2)",  icon: "▪", label: "Warehouse" },
  supplier:  { fill: "#F59E0B", glow: "rgba(245,158,11,0.2)",  icon: "◆", label: "Supplier"  },
  hub:       { fill: "#10B981", glow: "rgba(16,185,129,0.2)",  icon: "●", label: "Hub"       },
};

const ALGO_STYLE = {
  astar:    { stroke: "#E11D48", glow: "#E11D48", glowSpread: "rgba(225,29,72,0.15)",  label: "A* Route"  },
  // Retained as a defensive fallback style; the pipeline no longer emits Dijkstra.
  dijkstra: { stroke: "#7C3AED", glow: "#7C3AED", glowSpread: "rgba(124,58,237,0.15)", label: "A* Route" },
};

function AnimatedEdge({ x1, y1, x2, y2, algo, delay, id }) {
  const len = Math.hypot(x2 - x1, y2 - y1);
  const st = ALGO_STYLE[algo];
  return (
    <motion.line
      key={id}
      x1={x1} y1={y1} x2={x2} y2={y2}
      stroke={st.stroke}
      strokeWidth={3.5}
      strokeLinecap="round"
      filter={`url(#glow-${algo})`}
      strokeDasharray={len}
      strokeDashoffset={len}
      animate={{ strokeDashoffset: 0 }}
      transition={{ duration: 0.45, delay, ease: "easeOut" }}
    />
  );
}

function NodeTooltip({ node, x, y }) {
  if (!node) return null;
  const st = TYPE_STYLE[node.type] || TYPE_STYLE.hub;
  return (
    <g>
      <rect x={x + 12} y={y - 28} width={106} height={40}
        rx={8} fill="rgba(255,255,255,0.95)"
        stroke="var(--border)" strokeWidth={1} />
      <text x={x + 15} y={y - 12} fontSize={11} fontWeight={700} fill="var(--text-primary)"
        fontFamily="Inter, sans-serif">{node.name}</text>
      <text x={x + 15} y={y + 2} fontSize={9} fill={st.fill}
        fontFamily="Inter, sans-serif" textAnchor="start">
        {st.label} - ID {node.id}
      </text>
    </g>
  );
}

function LogisticsMap({ path, startNode, goalNode, activeAlgo }) {
  const [hovered, setHovered] = useState(null);

  const activeEdges = new Set();
  if (path?.length > 1) {
    for (let i = 0; i < path.length - 1; i++) {
      const key = [path[i], path[i + 1]].sort((a, b) => a - b).join("-");
      activeEdges.add(key);
    }
  }

  const pathSet = new Set(path || []);
  const hovNode = hovered != null ? NODE_POS[hovered] : null;

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      style={{
        width: "100%",
        borderRadius: 8,
        background: "#F8FAFC",
        border: "1px solid var(--border)",
        cursor: "crosshair",
      }}
    >
      <defs>
        <filter id="glow-astar" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="glow-dijkstra" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <radialGradient id="bg-grad" cx="50%" cy="50%" r="70%">
          <stop offset="0%"   stopColor="rgba(255,255,255,0)" />
          <stop offset="100%" stopColor="rgba(241,245,249,0.5)" />
        </radialGradient>
      </defs>

      <rect width={SVG_W} height={SVG_H} fill="url(#bg-grad)" rx={16} />

      {/* Grid lines (subtle) */}
      {[...Array(6)].map((_, i) => (
        <line key={`gl${i}`}
          x1={PAD_X + i * ((SVG_W - 2*PAD_X) / 5)} y1={PAD_Y}
          x2={PAD_X + i * ((SVG_W - 2*PAD_X) / 5)} y2={SVG_H - PAD_Y}
          stroke="rgba(0,0,0,0.03)" strokeWidth={1}
        />
      ))}
      {[...Array(5)].map((_, i) => (
        <line key={`gr${i}`}
          x1={PAD_X} y1={PAD_Y + i * ((SVG_H - 2*PAD_Y) / 4)}
          x2={SVG_W - PAD_X} y2={PAD_Y + i * ((SVG_H - 2*PAD_Y) / 4)}
          stroke="rgba(0,0,0,0.03)" strokeWidth={1}
        />
      ))}

      {/* Background edges */}
      {EDGES.map(([a, b]) => {
        const pa = NODE_POS[a], pb = NODE_POS[b];
        const key = [a, b].sort((x, y) => x - y).join("-");
        const isActive = activeEdges.has(key);
        return (
          !isActive && (
            <line key={`bg-${key}`}
              x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
              stroke="rgba(0,0,0,0.08)"
              strokeWidth={1.5}
              strokeLinecap="round"
            />
          )
        );
      })}

      {/* Active path edges */}
      {path?.length > 1 && path.slice(0, -1).map((nodeId, i) => {
        const pa = NODE_POS[nodeId];
        const pb = NODE_POS[path[i + 1]];
        return (
          <AnimatedEdge
            key={`path-${activeAlgo}-${i}`}
            id={`path-${activeAlgo}-${i}`}
            x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
            algo={activeAlgo}
            delay={i * 0.18}
          />
        );
      })}

      {/* Nodes */}
      {Object.entries(NODE_POS).map(([idStr, node]) => {
        const id = Number(idStr);
        const st = TYPE_STYLE[node.type] || TYPE_STYLE.hub;
        const isActive = pathSet.has(id);
        const isStart = id === startNode;
        const isGoal  = id === goalNode;
        
        let activeColor = st.fill;
        if (isActive && activeAlgo) {
          activeColor = ALGO_STYLE[activeAlgo].stroke;
        }

        const r = isStart || isGoal ? 10 : isActive ? 8 : 6;

        return (
          <g
            key={id}
            style={{ cursor: "pointer" }}
            onMouseEnter={() => setHovered(id)}
            onMouseLeave={() => setHovered(null)}
          >
            {(isActive || isStart || isGoal) && (
              <motion.circle
                cx={node.x} cy={node.y} r={r + 8}
                fill={isStart ? "rgba(251,191,36,0.15)"
                  : isGoal ? "rgba(52,211,153,0.15)"
                  : ALGO_STYLE[activeAlgo]?.glowSpread || "rgba(0,0,0,0)"}
                initial={{ r: r + 5, opacity: 0 }}
                animate={{ r: [r + 5, r + 12, r + 5], opacity: [0.6, 0.2, 0.6] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              />
            )}

            <motion.circle
              cx={node.x} cy={node.y}
              r={r}
              fill={isStart ? "#FBBF24" : isGoal ? "#34D399" : isActive ? activeColor : st.fill}
              stroke={isStart || isGoal ? "white" : isActive ? activeColor : "rgba(255,255,255,0.8)"}
              strokeWidth={isStart || isGoal ? 2.5 : isActive ? 2 : 1.5}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3, delay: id * 0.03, type: "spring", stiffness: 300 }}
            />

            {isStart && (
              <text x={node.x} y={node.y - r - 6}
                textAnchor="middle" fontSize={9} fontWeight={800}
                fill="#F59E0B" fontFamily="Inter, sans-serif">START</text>
            )}
            {isGoal && (
              <text x={node.x} y={node.y - r - 6}
                textAnchor="middle" fontSize={9} fontWeight={800}
                fill="#10B981" fontFamily="Inter, sans-serif">GOAL</text>
            )}

            <text
              x={node.x}
              y={node.y + r + 12}
              textAnchor="middle"
              fontSize={isActive || isStart || isGoal ? 10 : 8.5}
              fontWeight={isActive || isStart || isGoal ? 700 : 500}
              fill={isStart ? "#F59E0B" : isGoal ? "#10B981"
                : isActive ? activeColor : "var(--text-dim)"}
              fontFamily="Inter, sans-serif"
            >
              {node.name}
            </text>
          </g>
        );
      })}

      <AnimatePresence>
        {hovered != null && hovNode && (
          <NodeTooltip
            node={{ ...hovNode, id: hovered }}
            x={hovNode.x} y={hovNode.y}
          />
        )}
      </AnimatePresence>
    </svg>
  );
}

function PathBreadcrumb({ result, algo }) {
  if (!result || result.error || !result.path_names?.length) return null;
  const st = ALGO_STYLE[algo];
  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
      style={{
        padding: "16px 20px",
        background: `linear-gradient(135deg, ${st.glowSpread}, #FFFFFF)`,
        border: `1px solid ${st.stroke}33`,
        borderRadius: 16, marginTop: 16,
        boxShadow: `0 4px 15px ${st.glowSpread}`
      }}
    >
      <p style={{ fontSize: 11, fontWeight: 800, color: st.stroke,
        textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }}>
        {"🚀 A* Optimal Path"}
        <span style={{ float: "right", fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-primary)" }}>
          Cost: {result.total_cost} - {result.nodes_explored} nodes explored
        </span>
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 6 }}>
        {result.path_names.map((name, i) => {
          const city = name.includes("-") ? name.split("-").slice(1).join("-") : name;
          return (
            <Fragment key={i}>
              <motion.span
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.08 }}
                style={{
                  padding: "4px 12px", borderRadius: 8,
                  background: i === 0 ? "rgba(251,191,36,0.15)"
                    : i === result.path_names.length - 1 ? "rgba(52,211,153,0.15)"
                    : `${st.stroke}15`,
                  border: `1px solid ${i === 0 ? "#FBBF2444" : i === result.path_names.length - 1 ? "#34D39944" : st.stroke + "33"}`,
                  fontSize: 12, fontWeight: 700,
                  color: i === 0 ? "#D97706" : i === result.path_names.length - 1 ? "#059669" : st.stroke,
                }}
              >
                {city}
              </motion.span>
              {i < result.path_names.length - 1 && (
                <span style={{ color: `${st.stroke}60`, fontSize: 16, fontWeight: 800 }}>→</span>
              )}
            </Fragment>
          );
        })}
      </div>
    </motion.div>
  );
}

export default function RouteOptimization({ liveOrder }) {
  const routeResult = liveOrder?.route;
  
  // Pipeline standardized on A*; default to it whenever a route is present.
  const activeAlgo = routeResult?.path?.length ? "astar" : null;

  const startNode = routeResult?.path ? routeResult.path[0] : null;
  const goalNode = routeResult?.path ? routeResult.path[routeResult.path.length - 1] : null;

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
      <div className="page-header">
        <div className="page-badge"><Network size={10} /> Graph Search Algorithms</div>
        <h1 className="page-title">Route Optimizer</h1>
        <p className="page-subtitle">
          Automated routing output for the current live order using A* heuristic shortest-path search.
        </p>
      </div>

      <div className="route-layout">
        {/* Map */}
        <div className="card" style={{ padding: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div>
              <p className="card-label">Logistics Network</p>
              <p className="card-title" style={{ fontSize: 15, marginBottom: 2 }}>18 Indian Cities - 28 Edges</p>
            </div>
            <div style={{ display: "flex", gap: 12, fontSize: 12, color: "var(--text-muted)", fontWeight: 500 }}>
              {Object.entries(TYPE_STYLE).map(([type, s]) => (
                <span key={type}>
                  <span style={{ color: s.fill, fontWeight: 700 }}>●</span> {s.label}
                </span>
              ))}
            </div>
          </div>

          <LogisticsMap
            key={JSON.stringify(routeResult?.path || [])} // Re-animate on new path
            path={routeResult?.path}
            startNode={startNode}
            goalNode={goalNode}
            activeAlgo={activeAlgo}
          />

          <div style={{ display: "flex", gap: 24, marginTop: 16, fontSize: 12, color: "var(--text-muted)", flexWrap: "wrap", fontWeight: 500 }}>
            <span><span style={{ color: "#F59E0B", fontWeight: 700 }}>●</span> Start</span>
            <span><span style={{ color: "#10B981", fontWeight: 700 }}>●</span> Goal</span>
            <span><span style={{ color: "#E11D48", fontWeight: 700 }}>─</span> A* Route</span>
          </div>
        </div>

        {/* Results panel */}
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          <p className="card-label">Route Results</p>
          <p className="card-title" style={{ fontSize: 15, marginBottom: 4 }}>Live Order Routing</p>

          <AnimatePresence>
            {!routeResult && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="loading-center"
                style={{
                  flex: 1, border: "1px dashed var(--border)",
                  borderRadius: 16, margin: "16px 0", minHeight: 200,
                  background: "var(--bg-card)"
                }}
              >
                <Network size={36} style={{ color: "var(--text-dim)", marginBottom: 12 }} />
                <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Process a live order to see the route</span>
              </motion.div>
            )}
          </AnimatePresence>

          {routeResult && activeAlgo && (
            <PathBreadcrumb result={routeResult} algo={activeAlgo} />
          )}

          {routeResult?.error && (
            <div className="alert alert-danger" style={{ marginTop: 16 }}>{routeResult.error}</div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
