import math

# ─── 18-Node Logistics Graph ──────────────────────────────────────────────────
# Each node: (name, latitude, longitude, type)
NODES = {
    0:  ("Warehouse-Mumbai",    19.0760,  72.8777, "warehouse"),
    1:  ("Warehouse-Delhi",     28.6139,  77.2090, "warehouse"),
    2:  ("Warehouse-Chennai",   13.0827,  80.2707, "warehouse"),
    3:  ("Supplier-Pune",       18.5204,  73.8567, "supplier"),
    4:  ("Supplier-Ahmedabad",  23.0225,  72.5714, "supplier"),
    5:  ("Supplier-Hyderabad",  17.3850,  78.4867, "supplier"),
    6:  ("Hub-Bengaluru",       12.9716,  77.5946, "hub"),
    7:  ("Hub-Kolkata",         22.5726,  88.3639, "hub"),
    8:  ("Hub-Jaipur",          26.9124,  75.7873, "hub"),
    9:  ("Hub-Surat",           21.1702,  72.8311, "hub"),
    10: ("Supplier-Nagpur",     21.1458,  79.0882, "supplier"),
    11: ("Warehouse-Kochi",      9.9312,  76.2673, "warehouse"),
    12: ("Hub-Indore",          22.7196,  75.8577, "hub"),
    13: ("Supplier-Coimbatore", 11.0168,  76.9558, "supplier"),
    14: ("Hub-Bhopal",          23.2599,  77.4126, "hub"),
    15: ("Supplier-Lucknow",    26.8467,  80.9462, "supplier"),
    16: ("Hub-Chandigarh",      30.7333,  76.7794, "hub"),
    17: ("Warehouse-Guwahati",  26.1445,  91.7362, "warehouse"),
}

# Edges: (node_a, node_b, distance_km, traffic_factor, capacity)
EDGES = [
    (0,  3,   148, 1.2, 100),
    (0,  6,   984, 1.1, 200),
    (0,  9,   274, 1.3, 150),
    (1,  8,   268, 1.1, 180),
    (1,  15,  511, 1.2, 120),
    (1,  16,  248, 1.0, 200),
    (2,  5,   626, 1.1, 150),
    (2,  13,  508, 1.2, 100),
    (3,  6,   840, 1.3, 120),
    (3,  9,   126, 1.2, 180),
    (4,  9,   265, 1.1, 200),
    (4,  8,   668, 1.2, 150),
    (5,  6,   570, 1.0, 200),
    (5,  10,  699, 1.3, 120),
    (6,  11,  914, 1.1, 180),
    (6,  13,  358, 1.0, 150),
    (7,  17,  584, 1.4, 100),
    (8,  12,  483, 1.1, 200),
    (8,  14,  445, 1.0, 180),
    (9,  12,  448, 1.2, 150),
    (10, 14,  492, 1.1, 180),
    (11, 13,  215, 1.0, 200),
    (12, 14,  185, 1.0, 200),
    (14, 15,  552, 1.2, 120),
    (15, 16,  518, 1.1, 180),
    (16, 17,  871, 1.3, 120),
    (1,  7,  1495, 1.4, 100),
    
    # Parallel "Air Freight" edges between major hubs
    (8,  12,  483, 0.1, 50, "air"),
    (8,  14,  445, 0.1, 50, "air"),
    (9,  12,  448, 0.1, 50, "air"),
    (12, 14,  185, 0.1, 50, "air"),
]


def haversine(n1: int, n2: int) -> float:
    """Great-circle distance in km between two nodes."""
    _, lat1, lon1, _ = NODES[n1]
    _, lat2, lon2, _ = NODES[n2]
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_adjacency(edges=None, mode: str = "time") -> dict:
    """
    Build adjacency list from EDGES.
    mode='time'  → weight = distance × traffic_factor  (A* emergency)
    mode='cost'  → weight = fuel_cost + time_cost      (Standard cost routing)
    """
    if edges is None:
        edges = EDGES
    adj = {n: [] for n in NODES}
    for edge in edges:
        if len(edge) == 6:
            a, b, dist, traffic, cap, transport_mode = edge
        else:
            a, b, dist, traffic, cap = edge
            transport_mode = "ground"

        if mode == "time":
            w = dist * traffic
        else:
            fuel_cost = dist * 0.08
            time_cost = dist * traffic * 0.5
            w = fuel_cost + time_cost
            if transport_mode == "air":
                w *= 10.0  # Air freight is extremely expensive

        adj[a].append((b, round(w, 2)))
        adj[b].append((a, round(w, 2)))
    return adj


def get_graph_data() -> dict:
    """Return nodes + edges as JSON-serialisable dicts for the frontend."""
    nodes = [
        {"id": nid, "name": data[0], "lat": data[1], "lon": data[2], "type": data[3]}
        for nid, data in NODES.items()
    ]
    edges_list = []
    for edge in EDGES:
        if len(edge) == 6:
            a, b, dist, traffic, cap, transport_mode = edge
        else:
            a, b, dist, traffic, cap = edge
            transport_mode = "ground"
        
        edges_list.append({
            "source": a,
            "target": b,
            "distance": dist,
            "traffic": traffic,
            "capacity": cap,
            "mode": transport_mode
        })
    return {"nodes": nodes, "edges": edges_list}
