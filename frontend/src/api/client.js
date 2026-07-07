import axios from "axios";

const BASE = "http://localhost:8000/api";

const api = axios.create({ baseURL: BASE, timeout: 120000 });

export const getStatus = () => api.get("/status");
export const trainModels = (episodes = 10000) =>
  api.post("/train", { episodes });

export const getDemandForecast = (horizon = 7) =>
  api.get(`/demand/forecast?horizon=${horizon}`);

export const predictRisk = (features) => api.post("/risk/predict", features);

export const detectAnomalies = () => api.get("/anomaly/detect");
export const getSupplierScorecard = () => api.get("/supplier/scorecard");
export const checkSupplier = (metrics) => api.post("/anomaly/check", metrics);

export const getAstarRoute = (start, goal) =>
  api.post("/route/astar", { start, goal });
export const getDijkstraRoute = (start, goal) =>
  api.post("/route/dijkstra", { start, goal });

export const getRLAction = (params) => api.post("/rl/action", params);

export const getGraph = () => api.get("/graph");
export const getNodes = () => api.get("/nodes");

export const simulateOrder = () => api.get("/simulate/order");

export const runPipeline = (payload) => api.post("/pipeline/run", payload);

export const chatRL = (payload) => api.post("/rl/chat", payload);

export default api;
