import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from graph_construction import EDGES, NODES, build_adjacency


def dijkstra(start: int, goal: int = None) -> dict:
    """
    Dijkstra shortest-path search using cost-weighted graph edges.
    If goal is provided, returns {path, path_names, total_cost, nodes_explored, algorithm}.
    If goal is None, returns the cost map from start to every node.
    """
    adj = build_adjacency(EDGES, mode="cost")

    distances = {n: float("inf") for n in NODES}
    distances[start] = 0.0

    came_from = {}
    queue = [(0.0, start)]
    explored = 0
    closed = set()

    while queue:
        current_cost, current = heapq.heappop(queue)
        if current in closed:
            continue
        closed.add(current)
        explored += 1

        if goal is not None and current == goal:
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
                "total_cost": round(current_cost, 2),
                "nodes_explored": explored,
                "algorithm": "Dijkstra",
            }

        for neighbor, weight in adj[current]:
            if neighbor in closed:
                continue
            tentative = current_cost + weight
            if tentative < distances[neighbor]:
                distances[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(queue, (tentative, neighbor))

    if goal is None:
        return {str(k): round(v, 2) for k, v in distances.items()}

    return {
        "path": [],
        "path_names": [],
        "error": "No path found",
        "algorithm": "Dijkstra",
    }


def standard_route(start_node: int, goal_node: int) -> dict:
    return dijkstra(start_node, goal_node)
