import heapq
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from graph_construction import NODES, EDGES, build_adjacency, haversine


def astar(start: int, goal: int) -> dict:
    """
    A* search using Haversine as admissible heuristic.
    Returns: {path, path_names, total_cost, nodes_explored, algorithm}
    """
    adj = build_adjacency(EDGES, mode="time")

    open_set = []
    heapq.heappush(open_set, (0.0, start))

    g = {n: float("inf") for n in NODES}
    g[start] = 0.0

    came_from: dict = {}
    explored = 0
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        explored += 1

        if current == goal:
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return {
                "path": path,
                "path_names": [NODES[n][0] for n in path],
                "total_cost": round(g[goal], 2),
                "nodes_explored": explored,
                "algorithm": "A*",
            }

        for neighbor, weight in adj[current]:
            if neighbor in closed:
                continue
            tentative_g = g[current] + weight
            if tentative_g < g[neighbor]:
                came_from[neighbor] = current
                g[neighbor] = tentative_g
                h = haversine(neighbor, goal)
                f = tentative_g + h
                heapq.heappush(open_set, (f, neighbor))

    return {"path": [], "path_names": [], "error": "No path found", "algorithm": "A*"}


def emergency_route(start_node: int, goal_node: int, trigger_reason: str = "") -> dict:
    result = astar(start_node, goal_node)
    result["trigger"] = trigger_reason
    return result
