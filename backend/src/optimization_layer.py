"""
optimization_layer.py — SmartChain AI "Layer 4.5" Optimization Layer

Bridges the RL agent's macro-decision (e.g. REORDER_LARGE, SWITCH_SUPPLIER)
with a physical inventory-allocation solver. The RL agent decides *what* to do;
this layer decides *how* to physically fulfil it at minimum shipping cost using a
linear program (GLOP / ortools).

Model
-----
Decision variables : units_shipped[warehouse, region]  (continuous, >= 0)
Objective          : minimize  sum(units_shipped * cost_per_unit)
Supply constraints : sum over regions of a warehouse's shipments <= its capacity
Demand constraints : sum over warehouses into a region        >= that region's demand

The RL macro-decision enters through `action_context`, which can dynamically
override warehouse capacities before the solve (e.g. SWITCH_SUPPLIER zeroes out
the primary warehouse so the solver reroutes through the alternatives).
"""

from ortools.linear_solver import pywraplp


# ─── Default network definition ───────────────────────────────────────────────
# Warehouses and their baseline capacities (units available to ship).
DEFAULT_WAREHOUSES = {
    "primary":   1000.0,
    "secondary":  700.0,
    "tertiary":   500.0,
}

# Regions receive a fixed share of the total forecast demand.
DEFAULT_REGION_SHARES = {
    "north": 0.40,
    "south": 0.35,
    "east":  0.25,
}

# Per-unit shipping cost from each warehouse to each region.
DEFAULT_COST_MATRIX = {
    "primary":   {"north": 2.0, "south": 3.5, "east": 4.0},
    "secondary": {"north": 4.5, "south": 2.5, "east": 3.0},
    "tertiary":  {"north": 5.0, "south": 4.0, "east": 2.0},
}


def _apply_action_context(capacities: dict, action_context) -> dict:
    """
    Dynamically adjust warehouse capacities based on the RL macro-decision.

    Currently handled:
      SWITCH_SUPPLIER -> primary warehouse is taken offline (capacity 0), forcing
                         the solver to fulfil demand from secondary/tertiary.

    Unknown / None contexts leave capacities untouched. Returns a new dict so the
    module-level defaults are never mutated.
    """
    caps = dict(capacities)
    if action_context == "SWITCH_SUPPLIER":
        if "primary" in caps:
            caps["primary"] = 0.0
    return caps


def _total_demand(demand_forecast) -> float:
    """Accept either a per-day list/array or a single scalar and return the total."""
    if demand_forecast is None:
        return 0.0
    if isinstance(demand_forecast, (int, float)):
        return float(demand_forecast)
    try:
        return float(sum(float(v) for v in demand_forecast))
    except TypeError:
        return float(demand_forecast)


def optimize_warehouse_allocation(
    demand_forecast,
    action_context=None,
    warehouses: dict = None,
    region_shares: dict = None,
    cost_matrix: dict = None,
) -> dict:
    """
    Solve the minimum-cost warehouse→region allocation for the forecast demand.

    Parameters
    ----------
    demand_forecast : list[float] | float
        Forecast demand (e.g. XGBoost 7-day forecast). A list is summed to a
        total; a scalar is used directly.
    action_context : str | None
        RL macro-decision that can reshape the problem (e.g. "SWITCH_SUPPLIER").
    warehouses, region_shares, cost_matrix : dict, optional
        Override the default network definition. Defaults are used when omitted.

    Returns
    -------
    dict
        {
          "status": "OPTIMAL" | "INFEASIBLE" | "SOLVER_UNAVAILABLE" | "ABNORMAL",
          "total_optimized_cost": float,
          "allocations": [
              {"warehouse": str, "region": str, "units": float,
               "cost_per_unit": float, "cost": float},
              ...
          ],
        }
    """
    warehouses    = warehouses    if warehouses    is not None else DEFAULT_WAREHOUSES
    region_shares = region_shares if region_shares is not None else DEFAULT_REGION_SHARES
    cost_matrix   = cost_matrix   if cost_matrix   is not None else DEFAULT_COST_MATRIX

    # Dynamically update capacities from the RL macro-decision.
    capacities = _apply_action_context(warehouses, action_context)

    # Split total demand into per-region requirements.
    total = _total_demand(demand_forecast)
    region_demand = {r: total * share for r, share in region_shares.items()}

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        return {"status": "SOLVER_UNAVAILABLE", "total_optimized_cost": 0.0, "allocations": []}

    # Decision variables: units shipped from each warehouse to each region.
    x = {}
    for w in capacities:
        for r in region_demand:
            x[(w, r)] = solver.NumVar(0.0, solver.infinity(), f"x_{w}_{r}")

    # Supply constraints: a warehouse cannot ship more than its capacity.
    for w, cap in capacities.items():
        solver.Add(solver.Sum(x[(w, r)] for r in region_demand) <= cap)

    # Regional demand constraints: each region must be fully served.
    for r, dem in region_demand.items():
        solver.Add(solver.Sum(x[(w, r)] for w in capacities) >= dem)

    # Objective: minimize total shipping cost.
    solver.Minimize(
        solver.Sum(
            x[(w, r)] * cost_matrix[w][r]
            for w in capacities
            for r in region_demand
        )
    )

    result = solver.Solve()

    if result == pywraplp.Solver.INFEASIBLE:
        return {"status": "INFEASIBLE", "total_optimized_cost": 0.0, "allocations": []}
    if result not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {"status": "ABNORMAL", "total_optimized_cost": 0.0, "allocations": []}

    allocations = []
    for (w, r), var in x.items():
        units = var.solution_value()
        if units > 1e-6:  # skip zero-flow lanes
            unit_cost = cost_matrix[w][r]
            allocations.append({
                "warehouse":     w,
                "region":        r,
                "units":         round(units, 2),
                "cost_per_unit": unit_cost,
                "cost":          round(units * unit_cost, 2),
            })

    return {
        "status": "OPTIMAL",
        "total_optimized_cost": round(solver.Objective().Value(), 2),
        "allocations": allocations,
    }


if __name__ == "__main__":
    # Quick smoke test: normal operation vs. supplier switch.
    normal = optimize_warehouse_allocation([120, 130, 110, 140, 125, 135, 150])
    print("NORMAL:", normal["status"], normal["total_optimized_cost"])
    for a in normal["allocations"]:
        print("  ", a)

    switched = optimize_warehouse_allocation(
        [120, 130, 110, 140, 125, 135, 150], action_context="SWITCH_SUPPLIER"
    )
    print("SWITCH_SUPPLIER:", switched["status"], switched["total_optimized_cost"])
    for a in switched["allocations"]:
        print("  ", a)
