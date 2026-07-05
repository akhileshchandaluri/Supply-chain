import heapq
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from graph_construction import NODES, EDGES, build_adjacency


def dijkstra(start: int, goal: int = None) -> dict:
    """
    Dijkstra's algorithm for cost-optimal routing.
    If goal=None, returns cost map to all nodes.
    """
    adj = build_adjacency(EDGES, mode="cost")

    dist = {n: float("inf") for n in NODES}
    dist[start] = 0.0
    came_from: dict = {}
    visited: set = set()
    pq = [(0.0, start)]

    while pq:
        cost, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if goal is not None and current == goal:
            break

        for neighbor, weight in adj[current]:
            new_cost = cost + weight
            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    if goal is not None:
        if dist[goal] == float("inf"):
            return {
                "path": [],
                "path_names": [],
                "error": "No path found",
                "algorithm": "Dijkstra",
            }
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
            "total_cost": round(dist[goal], 2),
            "nodes_explored": len(visited),
            "algorithm": "Dijkstra",
        }

    # Return all-pairs cost map
    return {str(k): round(v, 2) for k, v in dist.items()}


def standard_route(start_node: int, goal_node: int) -> dict:
    return dijkstra(start_node, goal_node)
