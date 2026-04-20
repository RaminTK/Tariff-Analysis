"""
Misinformation-aware transportation + inventory model (Canada tariff rumor case)
------------------------------------------------------------------------------

This script builds and solves TWO models with Gurobi:

1) Nominal stochastic program (theta = 0): trusts nominal regime probabilities given a signal.
2) Distributionally robust model (theta > 0): hedges against misinformation via an L1 ambiguity set.

Key design choice for tractability:
- One signal s is observed for the whole planning horizon (12 months).
- One tariff regime ω realizes for the whole horizon.
This matches “rumor periods” as a planning environment, and keeps the model small enough to run fast.

You can later extend to rolling signals per month; the structure stays the same.

Run:
  python canada_tariff_dro.py
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import math

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    raise SystemExit("This script requires gurobipy. Install Gurobi + gurobipy first.") from e


# -----------------------------
# Case study configuration
# -----------------------------
@dataclass(frozen=True)
class CaseData:
    months: int = 12

    # Demand per month (units)
    dem_NE: int = 1000
    dem_MW: int = 800

    # Costs
    c_ship: float = 70.0      # transport cost per unit (Canada -> DC network)
    h: float = 4.0            # holding cost per unit per month
    p_short: float = 200.0    # shortage penalty per unit
    v: float = 300.0          # product value per unit for tariff calculation

    # Shipping capacity (units per month) - tune this for more/less stress
    U_month: int = 2000

    # Initial inventory at DCs
    I0_NJ: float = 0.0
    I0_IL: float = 0.0

    # DC assignment: NE served by NJ, MW served by IL (simple + realistic)
    # If you want flexibility, add variables to split shipments to both DCs.
    # Here: a(NE)=NJ, a(MW)=IL
    pass


def build_and_solve(theta: float, case: CaseData, verbose: bool = False) -> Dict:
    """
    Solve the DRO model with L1 ambiguity radius theta.
    theta=0 -> nominal expected value (no misinformation robustness).

    Returns a dict of results (objective, shipments, inventories, shortages, etc.).
    """
    T = list(range(1, case.months + 1))
    D = ["NJ", "IL"]
    R = ["NE", "MW"]

    # Tariff regimes (constant over horizon once realized)
    Omega = ["R0", "R1", "R2", "R3"]
    tau = {"R0": 0.00, "R1": 0.10, "R2": 0.35, "R3": 1.00}
    tariff_cost = {w: case.v * tau[w] for w in Omega}

    # Signals for the "rumor environment" (observed before decisions)
    S = ["N", "M", "E"]  # No-tariff, Moderate, Extreme
    Pr_s = {"N": 0.40, "M": 0.40, "E": 0.20}

    # Nominal regime probabilities conditional on signal
    # (You can calibrate these later from text / news cadence; these are sensible starting points.)
    qhat = {
        "N": {"R0": 0.85, "R1": 0.10, "R2": 0.05, "R3": 0.00},
        "M": {"R0": 0.25, "R1": 0.50, "R2": 0.20, "R3": 0.05},
        "E": {"R0": 0.05, "R1": 0.15, "R2": 0.30, "R3": 0.50},
    }

    # Demands
    dem = {(r, t): (case.dem_NE if r == "NE" else case.dem_MW) for r in R for t in T}
    assign = {"NE": "NJ", "MW": "IL"}  # fixed assignment

    # Pre-positioning procurement cost (optional)
    # If you want, set to something >0. Here 0 keeps the focus on holding vs shortage vs tariff.
    c_pre = {"NJ": 0.0, "IL": 0.0}

    # Build model
    m = gp.Model("canada_tariff_dro")
    m.Params.OutputFlag = 1 if verbose else 0

    # -----------------------------
    # Variables
    # -----------------------------
    # First stage: pre-position at start (added to t=1 inventory)
    x = m.addVars(D, lb=0.0, name="x")

    # Second stage variables indexed by signal s and regime w
    f = m.addVars(S, Omega, T, lb=0.0, ub=case.U_month, name="f")          # inbound shipments total
    g = m.addVars(S, Omega, D, T, lb=0.0, name="g")                         # inbound split to DCs
    I = m.addVars(S, Omega, D, T, lb=0.0, name="I")                         # inventory end of month
    y = m.addVars(S, Omega, R, T, lb=0.0, name="y")                         # deliveries
    z = m.addVars(S, Omega, R, T, lb=0.0, name="z")                         # shortages

    # Cost per (s,w): define as variable to connect robustification cleanly
    C = m.addVars(S, Omega, lb=0.0, name="C")

    # DRO epigraph variables per signal s
    eta = m.addVars(S, lb=-GRB.INFINITY, name="eta")  # robust expected recourse cost for signal s

    # Dual variables for inner max over q in L1 ball (per signal s)
    # alpha_{s,w} free, beta_s free, gamma_s >=0
    alpha = m.addVars(S, Omega, lb=-GRB.INFINITY, name="alpha")
    beta = m.addVars(S, lb=-GRB.INFINITY, name="beta")
    gamma = m.addVars(S, lb=0.0, name="gamma")

    # -----------------------------
    # Constraints
    # -----------------------------

    # Inbound split: sum_d g = f
    for s in S:
        for w in Omega:
            for t in T:
                m.addConstr(gp.quicksum(g[s, w, d, t] for d in D) == f[s, w, t],
                            name=f"inbound_split[{s},{w},{t}]")

    # Demand satisfaction: y + z = demand
    for s in S:
        for w in Omega:
            for r in R:
                for t in T:
                    m.addConstr(y[s, w, r, t] + z[s, w, r, t] == dem[(r, t)],
                                name=f"demand[{s},{w},{r},{t}]")

    # Inventory dynamics per DC with fixed region assignment
    I0 = {"NJ": case.I0_NJ, "IL": case.I0_IL}
    for s in S:
        for w in Omega:
            # t=1
            for d in D:
                outflow_t1 = gp.quicksum(y[s, w, r, 1] for r in R if assign[r] == d)
                m.addConstr(I[s, w, d, 1] == I0[d] + x[d] + g[s, w, d, 1] - outflow_t1,
                            name=f"inv_init[{s},{w},{d}]")
            # t>=2
            for t in T[1:]:
                for d in D:
                    outflow = gp.quicksum(y[s, w, r, t] for r in R if assign[r] == d)
                    m.addConstr(I[s, w, d, t] == I[s, w, d, t - 1] + g[s, w, d, t] - outflow,
                                name=f"inv[{s},{w},{d},{t}]")

    # Define cost C[s,w] as total horizon cost under (s,w)
    for s in S:
        for w in Omega:
            transport = gp.quicksum(case.c_ship * f[s, w, t] for t in T)
            tariffs = gp.quicksum(tariff_cost[w] * f[s, w, t] for t in T)
            holding = gp.quicksum(case.h * I[s, w, d, t] for d in D for t in T)
            shortage = gp.quicksum(case.p_short * z[s, w, r, t] for r in R for t in T)
            m.addConstr(C[s, w] == transport + tariffs + holding + shortage,
                        name=f"costdef[{s},{w}]")

    # -----------------------------
    # DRO robustification (exact via dual of inner maximization)
    # Robust constraint: eta_s >= max_{q in Q(s)} sum_w q_w * C[s,w]
    # Dual form:
    #   eta_s >= beta_s + sum_w alpha_{s,w} * qhat_{s,w} + gamma_s * theta
    #   alpha_{s,w} + beta_s >= C[s,w]
    #   -alpha_{s,w} + gamma_s >= 0
    #   alpha_{s,w} + gamma_s >= 0
    # -----------------------------
    for s in S:
        m.addConstr(
            eta[s] >= beta[s] + gp.quicksum(alpha[s, w] * qhat[s][w] for w in Omega) + gamma[s] * theta,
            name=f"eta_dual[{s}]"
        )
        for w in Omega:
            m.addConstr(alpha[s, w] + beta[s] >= C[s, w], name=f"dual1[{s},{w}]")
            m.addConstr(-alpha[s, w] + gamma[s] >= 0.0, name=f"dual2[{s},{w}]")
            m.addConstr(alpha[s, w] + gamma[s] >= 0.0, name=f"dual3[{s},{w}]")

    # -----------------------------
    # Objective: pre-positioning cost + expected robust recourse cost over signals
    # -----------------------------
    pre_cost = gp.quicksum(c_pre[d] * x[d] for d in D)
    exp_robust = gp.quicksum(Pr_s[s] * eta[s] for s in S)
    m.setObjective(pre_cost + exp_robust, GRB.MINIMIZE)

    # Optimize
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimization ended with status {m.Status}")

    # -----------------------------
    # Collect results
    # -----------------------------
    def val(v):  # safe float
        return float(v.X)

    results = {
        "theta": theta,
        "objective": float(m.ObjVal),
        "x": {d: val(x[d]) for d in D},
        "eta": {s: val(eta[s]) for s in S},
        "C": {(s, w): val(C[s, w]) for s in S for w in Omega},
        "shipments_total": {(s, w): sum(val(f[s, w, t]) for t in T) for s in S for w in Omega},
        "shortage_total": {(s, w): sum(val(z[s, w, r, t]) for r in R for t in T) for s in S for w in Omega},
        "inventory_end": {(s, w, d): val(I[s, w, d, T[-1]]) for s in S for w in Omega for d in D},
    }

    # Add some interpretable breakdowns for one representative signal (Moderate)
    s0 = "M"
    breakdown = {}
    for w in Omega:
        ship = sum(val(f[s0, w, t]) for t in T)
        avg_monthly_ship = ship / len(T)
        breakdown[w] = {
            "total_ship": ship,
            "avg_monthly_ship": avg_monthly_ship,
            "total_shortage": sum(val(z[s0, w, r, t]) for r in R for t in T),
            "end_inv_NJ": val(I[s0, w, "NJ", T[-1]]),
            "end_inv_IL": val(I[s0, w, "IL", T[-1]]),
            "scenario_cost": val(C[s0, w]),
        }
    results["moderate_signal_summary"] = breakdown

    return results


def pretty_print(res: Dict):
    print("\n" + "=" * 78)
    print(f"RESULTS (theta={res['theta']})")
    print("=" * 78)
    print(f"Objective value: {res['objective']:.2f}")
    print("Pre-position x:")
    for d, v in res["x"].items():
        print(f"  x[{d}] = {v:.2f}")
    print("Robust epigraph eta (per signal):")
    for s, v in res["eta"].items():
        print(f"  eta[{s}] = {v:.2f}")

    print("\nModerate-signal (s='M') scenario summary by regime ω:")
    for w, info in res["moderate_signal_summary"].items():
        print(f"  Regime {w}:")
        print(f"    total_ship = {info['total_ship']:.2f}  (avg/month {info['avg_monthly_ship']:.2f})")
        print(f"    total_shortage = {info['total_shortage']:.2f}")
        print(f"    end_inv_NJ = {info['end_inv_NJ']:.2f}, end_inv_IL = {info['end_inv_IL']:.2f}")
        print(f"    scenario_cost = {info['scenario_cost']:.2f}")


if __name__ == "__main__":
    case = CaseData()

    # 1) Nominal model (no misinformation robustness)
    res_nominal = build_and_solve(theta=0.0, case=case, verbose=False)
    pretty_print(res_nominal)

    # 2) Robust model (misinformation-aware)
    res_robust = build_and_solve(theta=0.30, case=case, verbose=False)
    pretty_print(res_robust)

    # Quick comparison headline
    print("\n" + "-" * 78)
    print("Headline comparison:")
    print(f"  Nominal objective (theta=0.00): {res_nominal['objective']:.2f}")
    print(f"  Robust  objective (theta=0.30): {res_robust['objective']:.2f}")
    print("-" * 78)