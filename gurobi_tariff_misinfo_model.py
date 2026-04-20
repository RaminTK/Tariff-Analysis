
"""
Two-stage stochastic Gurobi model for supply chain planning under tariff uncertainty and misinformation.

Inputs are read from an Excel workbook with sheets:
- Suppliers
- Warehouses
- DemandZones
- InboundCosts
- OutboundCosts
- Signals
- TariffScenarios
- ConditionalProb

Optional generator input sheets:
- TariffPriors
- SignalDesign

Run:
    python gurobi_tariff_misinfo_model.py dataset_template.xlsx

Outputs:
- solver_summary.json
- x_solution.csv
- y_solution.csv
- r_solution.csv
- z_solution.csv
- I_solution.csv
- realized_costs_by_signal_scenario.csv
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:
    raise ImportError(
        "gurobipy is required to run this file. Install Gurobi and gurobipy, "
        "then activate your license before running."
    ) from exc


REQUIRED_SHEETS = [
    "Suppliers",
    "Warehouses",
    "DemandZones",
    "InboundCosts",
    "OutboundCosts",
    "Signals",
    "TariffScenarios",
    "ConditionalProb",
]


def _read_sheet(xl: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xl, sheet_name=sheet)
    df = df.dropna(how="all")
    # Keep only first contiguous block of non-empty columns if users leave notes to the right
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_data(workbook_path: str | Path) -> dict:
    xl = pd.ExcelFile(workbook_path)

    missing = [s for s in REQUIRED_SHEETS if s not in xl.sheet_names]
    if missing:
        raise ValueError(f"Workbook is missing required sheets: {missing}")

    suppliers = _read_sheet(xl, "Suppliers")
    warehouses = _read_sheet(xl, "Warehouses")
    demandzones = _read_sheet(xl, "DemandZones")
    inbound = _read_sheet(xl, "InboundCosts")
    outbound = _read_sheet(xl, "OutboundCosts")
    signals = _read_sheet(xl, "Signals")
    tariff = _read_sheet(xl, "TariffScenarios")
    cond = _read_sheet(xl, "ConditionalProb")

    # Standardize ids as strings
    for df, cols in [
        (suppliers, ["supplier_id"]),
        (warehouses, ["warehouse_id"]),
        (demandzones, ["zone_id"]),
        (inbound, ["supplier_id", "warehouse_id"]),
        (outbound, ["warehouse_id", "zone_id"]),
        (signals, ["signal_id"]),
        (tariff, ["scenario_id"]),
        (cond, ["signal_id", "scenario_id"]),
    ]:
        for c in cols:
            df[c] = df[c].astype(str)

    S = suppliers["supplier_id"].tolist()
    W = warehouses["warehouse_id"].tolist()
    D = demandzones["zone_id"].tolist()
    M = signals["signal_id"].tolist()
    T = tariff["scenario_id"].tolist()

    p = suppliers.set_index("supplier_id")["procurement_cost"].to_dict()
    U = suppliers.set_index("supplier_id")["supply_capacity"].to_dict()

    I0 = warehouses.set_index("warehouse_id")["initial_inventory"].to_dict()
    Cap = warehouses.set_index("warehouse_id")["capacity"].to_dict()
    h = warehouses.set_index("warehouse_id")["holding_cost"].to_dict()

    d = demandzones.set_index("zone_id")["demand"].to_dict()
    pi = demandzones.set_index("zone_id")["shortage_penalty"].to_dict()

    # full pair dictionaries
    c_in = {(row["supplier_id"], row["warehouse_id"]): float(row["inbound_cost"]) for _, row in inbound.iterrows()}
    c_reg = {(row["warehouse_id"], row["zone_id"]): float(row["regular_cost"]) for _, row in outbound.iterrows()}
    e = {(row["warehouse_id"], row["zone_id"]): float(row["emergency_cost"]) for _, row in outbound.iterrows()}

    P = signals.set_index("signal_id")["signal_prob"].to_dict()
    tau_hat = signals.set_index("signal_id")["perceived_tariff"].to_dict()

    tau = tariff.set_index("scenario_id")["actual_tariff"].to_dict()
    Q = {(row["signal_id"], row["scenario_id"]): float(row["cond_prob"]) for _, row in cond.iterrows()}

    # Validation
    missing_inbound = [(i, j) for i in S for j in W if (i, j) not in c_in]
    if missing_inbound:
        raise ValueError(f"InboundCosts is missing supplier-warehouse pairs, first few: {missing_inbound[:10]}")

    missing_outbound = [(j, k) for j in W for k in D if (j, k) not in c_reg or (j, k) not in e]
    if missing_outbound:
        raise ValueError(f"OutboundCosts is missing warehouse-zone pairs, first few: {missing_outbound[:10]}")

    missing_cond = [(m, s) for m in M for s in T if (m, s) not in Q]
    if missing_cond:
        raise ValueError(f"ConditionalProb is missing signal-scenario pairs, first few: {missing_cond[:10]}")

    tol = 1e-6
    if abs(sum(P[m] for m in M) - 1.0) > tol:
        raise ValueError(f"Signal probabilities must sum to 1.0, found {sum(P[m] for m in M):.6f}")

    for m in M:
        row_sum = sum(Q[m, s] for s in T)
        if abs(row_sum - 1.0) > tol:
            raise ValueError(f"Conditional probabilities for signal {m} must sum to 1.0, found {row_sum:.6f}")

    return {
        "S": S, "W": W, "D": D, "M": M, "T": T,
        "p": p, "U": U, "I0": I0, "Cap": Cap, "h": h,
        "d": d, "pi": pi, "c_in": c_in, "c_reg": c_reg, "e": e,
        "P": P, "tau_hat": tau_hat, "tau": tau, "Q": Q,
    }


def build_model(data: dict) -> gp.Model:
    S, W, D, M, T = data["S"], data["W"], data["D"], data["M"], data["T"]
    p, U, I0, Cap, h = data["p"], data["U"], data["I0"], data["Cap"], data["h"]
    d, pi, c_in, c_reg, e = data["d"], data["pi"], data["c_in"], data["c_reg"], data["e"]
    P, tau_hat, Q = data["P"], data["tau_hat"], data["Q"]

    model = gp.Model("tariff_misinformation_supply_chain")

    # Main first-stage decision
    x = model.addVars(S, W, M, lb=0.0, name="x")

    # Second-stage recourse and accounting variables
    y = model.addVars(W, D, M, T, lb=0.0, name="y")
    r = model.addVars(W, D, M, T, lb=0.0, name="r")
    z = model.addVars(D, M, T, lb=0.0, name="z")
    I = model.addVars(W, M, T, lb=0.0, name="I")

    # Objective: first-stage cost + expected recourse cost
    first_stage = gp.quicksum(
        P[m] * (p[i] * (1.0 + tau_hat[m]) + c_in[i, j]) * x[i, j, m]
        for i in S for j in W for m in M
    )

    recourse = gp.quicksum(
        P[m] * Q[m, s] * (
            gp.quicksum(c_reg[j, k] * y[j, k, m, s] for j in W for k in D) +
            gp.quicksum(e[j, k] * r[j, k, m, s] for j in W for k in D) +
            gp.quicksum(pi[k] * z[k, m, s] for k in D) +
            gp.quicksum(h[j] * I[j, m, s] for j in W)
        )
        for m in M for s in T
    )

    model.setObjective(first_stage + recourse, GRB.MINIMIZE)

    # Supplier capacity
    model.addConstrs(
        (gp.quicksum(x[i, j, m] for j in W) <= U[i]
         for i in S for m in M),
        name="supplier_capacity"
    )

    # Warehouse balance
    model.addConstrs(
        (
            gp.quicksum(x[i, j, m] for i in S) + I0[j]
            == gp.quicksum(y[j, k, m, s] + r[j, k, m, s] for k in D) + I[j, m, s]
            for j in W for m in M for s in T
        ),
        name="warehouse_balance"
    )

    # Warehouse capacity
    model.addConstrs(
        (I[j, m, s] <= Cap[j] for j in W for m in M for s in T),
        name="warehouse_capacity"
    )

    # Demand satisfaction
    model.addConstrs(
        (
            gp.quicksum(y[j, k, m, s] + r[j, k, m, s] for j in W) + z[k, m, s]
            == d[k]
            for k in D for m in M for s in T
        ),
        name="demand_balance"
    )

    model._vars = {"x": x, "y": y, "r": r, "z": z, "I": I}
    return model


def extract_solution(model: gp.Model, data: dict) -> dict[str, pd.DataFrame]:
    S, W, D, M, T = data["S"], data["W"], data["D"], data["M"], data["T"]
    p, P, tau, c_in, c_reg, e, pi, h, Q = (
        data["p"], data["P"], data["tau"], data["c_in"], data["c_reg"], data["e"], data["pi"], data["h"], data["Q"]
    )
    x, y, r, z, I = model._vars["x"], model._vars["y"], model._vars["r"], model._vars["z"], model._vars["I"]

    x_rows = []
    for i in S:
        for j in W:
            for m in M:
                val = x[i, j, m].X
                if val > 1e-8:
                    x_rows.append({"supplier_id": i, "warehouse_id": j, "signal_id": m, "x": val})
    y_rows = []
    for j in W:
        for k in D:
            for m in M:
                for s in T:
                    val = y[j, k, m, s].X
                    if val > 1e-8:
                        y_rows.append({"warehouse_id": j, "zone_id": k, "signal_id": m, "scenario_id": s, "y": val})
    r_rows = []
    for j in W:
        for k in D:
            for m in M:
                for s in T:
                    val = r[j, k, m, s].X
                    if val > 1e-8:
                        r_rows.append({"warehouse_id": j, "zone_id": k, "signal_id": m, "scenario_id": s, "r": val})
    z_rows = []
    for k in D:
        for m in M:
            for s in T:
                val = z[k, m, s].X
                if val > 1e-8:
                    z_rows.append({"zone_id": k, "signal_id": m, "scenario_id": s, "z": val})
    I_rows = []
    for j in W:
        for m in M:
            for s in T:
                val = I[j, m, s].X
                if val > 1e-8:
                    I_rows.append({"warehouse_id": j, "signal_id": m, "scenario_id": s, "I": val})

    # Realized cost by (m,s), replacing perceived tariff with actual tariff in first stage
    realized_rows = []
    for m in M:
        for s in T:
            first_realized = sum((p[i] * (1.0 + tau[s]) + c_in[i, j]) * x[i, j, m].X for i in S for j in W)
            second_realized = (
                sum(c_reg[j, k] * y[j, k, m, s].X for j in W for k in D) +
                sum(e[j, k] * r[j, k, m, s].X for j in W for k in D) +
                sum(pi[k] * z[k, m, s].X for k in D) +
                sum(h[j] * I[j, m, s].X for j in W)
            )
            realized_rows.append({
                "signal_id": m,
                "scenario_id": s,
                "signal_prob": P[m],
                "cond_prob": Q[m, s],
                "joint_prob": P[m] * Q[m, s],
                "realized_cost": first_realized + second_realized,
                "first_stage_realized_cost": first_realized,
                "second_stage_cost": second_realized,
            })

    return {
        "x": pd.DataFrame(x_rows),
        "y": pd.DataFrame(y_rows),
        "r": pd.DataFrame(r_rows),
        "z": pd.DataFrame(z_rows),
        "I": pd.DataFrame(I_rows),
        "realized_costs": pd.DataFrame(realized_rows),
    }


# def solve(workbook_path: str | Path, output_dir: str | Path | None = None) -> None:
#     workbook_path = Path(workbook_path)
#     output_dir = Path(output_dir) if output_dir else workbook_path.with_suffix("")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     data = load_data(workbook_path)
#     model = build_model(data)
#     model.optimize()

#     if model.Status != GRB.OPTIMAL:
#         raise RuntimeError(f"Model did not solve to optimality. Status code: {model.Status}")

#     solution = extract_solution(model, data)

#     summary = {
#         "status": int(model.Status),
#         "objective_value": float(model.ObjVal),
#         "num_variables": int(model.NumVars),
#         "num_constraints": int(model.NumConstrs),
#         "suppliers": len(data["S"]),
#         "warehouses": len(data["W"]),
#         "demand_zones": len(data["D"]),
#         "signals": len(data["M"]),
#         "tariff_scenarios": len(data["T"]),
#     }

#     with open(output_dir / "solver_summary.json", "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)

#     solution["x"].to_csv(output_dir / "x_solution.csv", index=False)
#     solution["y"].to_csv(output_dir / "y_solution.csv", index=False)
#     solution["r"].to_csv(output_dir / "r_solution.csv", index=False)
#     solution["z"].to_csv(output_dir / "z_solution.csv", index=False)
#     solution["I"].to_csv(output_dir / "I_solution.csv", index=False)
#     solution["realized_costs"].to_csv(output_dir / "realized_costs_by_signal_scenario.csv", index=False)

#     print(f"Solved. Objective value: {model.ObjVal:,.4f}")
#     print(f"Results written to: {output_dir}")


# if __name__ == "__main__":
#     # if len(sys.argv) < 2:
#     #     raise SystemExit("Usage: python gurobi_tariff_misinfo_model.py <input_workbook.xlsx> [output_dir]")
#     # wb = sys.argv[1]
#     # out = sys.argv[2] if len(sys.argv) >= 3 else None
#     # solve(wb, out)
#     wb = "dataset_template.xlsx"
#     out = None
#     solve(wb, out)



def solve(workbook_path: str | Path, output_dir: str | Path = "results") -> None:
    workbook_path = Path(workbook_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("RUN STARTED")
    print(f"Reading workbook from: {workbook_path}")
    print(f"Writing results to: {output_dir}")
    print("==================================================")

    data = load_data(workbook_path)
    model = build_model(data)
    model.optimize()

    print(f"Model status: {model.Status}")

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Model did not solve to optimality. Status code: {model.Status}")

    solution = extract_solution(model, data)

    summary = {
        "status": int(model.Status),
        "objective_value": float(model.ObjVal),
        "num_variables": int(model.NumVars),
        "num_constraints": int(model.NumConstrs),
        "suppliers": len(data["S"]),
        "warehouses": len(data["W"]),
        "demand_zones": len(data["D"]),
        "signals": len(data["M"]),
        "tariff_scenarios": len(data["T"]),
    }

    summary_path = output_dir / "solver_summary.json"
    x_path = output_dir / "x_solution.csv"
    y_path = output_dir / "y_solution.csv"
    r_path = output_dir / "r_solution.csv"
    z_path = output_dir / "z_solution.csv"
    I_path = output_dir / "I_solution.csv"
    realized_path = output_dir / "realized_costs_by_signal_scenario.csv"

    print("Now writing output files...")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    solution["x"].to_csv(x_path, index=False)
    solution["y"].to_csv(y_path, index=False)
    solution["r"].to_csv(r_path, index=False)
    solution["z"].to_csv(z_path, index=False)
    solution["I"].to_csv(I_path, index=False)
    solution["realized_costs"].to_csv(realized_path, index=False)

    print("Write complete.")
    print("Files now exist?")
    print("solver_summary.json:", summary_path.exists())
    print("x_solution.csv:", x_path.exists())
    print("y_solution.csv:", y_path.exists())
    print("r_solution.csv:", r_path.exists())
    print("z_solution.csv:", z_path.exists())
    print("I_solution.csv:", I_path.exists())
    print("realized_costs_by_signal_scenario.csv:", realized_path.exists())
    print(f"Solved. Objective value: {model.ObjVal:,.4f}")



if __name__ == "__main__":
    from pathlib import Path
    print("RUNNING SCRIPT:", Path(__file__).resolve())

    wb = "/Users/raminkhameneh/Library/CloudStorage/OneDrive-stevens.edu/tarrif analysis/dataset_template.xlsx"
    out = "/Users/raminkhameneh/Library/CloudStorage/OneDrive-stevens.edu/tarrif analysis/results"
    solve(wb, out)