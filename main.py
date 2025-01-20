import pandas as pd
from pyomo.common.enums import maximize
from pyomo.environ import (ConcreteModel, Var, Objective, Constraint, RangeSet, Binary, SolverFactory, value, Param)
from pyomo.opt import SolverStatus, TerminationCondition


def print_solution(model, results):
    """Prints the solution if optimal."""
    if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
        print("Optimal solution found.")
        if value(model.Split) == 0:
            print("Chosen strategy: 5-day maintenance.")
            for day in model.T:
                if value(model.Start5[day]) > 0.5:
                    print(f"5-day window starts on day {day}")
        else:
            print("Chosen strategy: 3-day + 2-day maintenance.")
            for day in model.T:
                if value(model.Start3[day]) > 0.5:
                    print(f"3-day window starts on day {day}")
                if value(model.Start2[day]) > 0.5:
                    print(f"2-day window starts on day {day}")

        print("Total Revenue:", round(value(model.profit), 2))
    else:
        print("No feasible solution or solver error.")


def build_model(prod, price, coeff, availability, capacity, fixed_cost):
    """Builds and returns a Pyomo model."""
    model = ConcreteModel()

    model.T = RangeSet(1, len(prod))
    model.Production = Param(model.T, initialize=prod)
    model.Prices = Param(model.T, initialize=price)
    model.MaintenanceCoeff = Param(model.T, initialize=coeff)
    model.Availability = Param(model.T, initialize=availability)

    model.Split = Var(domain=Binary)
    model.Maintenance = Var(model.T, domain=Binary)
    model.Start5 = Var(model.T, domain=Binary)
    model.Start3 = Var(model.T, domain=Binary)
    model.Start2 = Var(model.T, domain=Binary)

    for day in model.T:
        if not model.Availability[day]:
            model.Start5[day].fix(0)
            model.Start3[day].fix(0)
            model.Start2[day].fix(0)

    model.one_maintenance_start_5 = Constraint(expr=sum(model.Start5[day] for day in model.T) == 1 - model.Split)
    model.one_maintenance_start_3 = Constraint(expr=sum(model.Start3[day] for day in model.T) == model.Split)
    model.one_maintenance_start_2 = Constraint(expr=sum(model.Start2[day] for day in model.T) == model.Split)

    def link_maintenance_rule(m, day):
        days_5 = sum(m.Start5[k] for k in range(max(1, day - 4), day + 1))
        days_3 = sum(m.Start3[k] for k in range(max(1, day - 2), day + 1))
        days_2 = sum(m.Start2[k] for k in range(max(1, day - 1), day + 1))
        return m.Maintenance[day] == days_5 + days_3 + days_2

    model.link_maintenance = Constraint(model.T, rule=link_maintenance_rule)

    model.profit = Objective(
        expr=sum(
            capacity * model.Production[day] * model.Prices[day] * (1 - model.Maintenance[day]) -
            fixed_cost * model.MaintenanceCoeff[day] * model.Maintenance[day]
            for day in model.T
        ),
        sense=maximize
    )

    return model


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
excel_file = "Data.xlsx"

df_prod = pd.read_excel(excel_file, sheet_name="forcast production")
df_price = pd.read_excel(excel_file, sheet_name="price of electricity")
df_maint = pd.read_excel(excel_file, sheet_name="maintenance coefficient")

prod = {int(row['period']): float(row['forecastp']) for _, row in df_prod.iterrows()}
price = {int(row['period']): float(row['price']) for _, row in df_price.iterrows()}
coeff = {int(row['period']): float(row['coeff']) for _, row in df_maint.iterrows()}

T = max(prod.keys())
capacity = 20.0
fixed_cost = 500

# ---------------------------------------------------------------------
# Solve model without "no-maintenance window"
# ---------------------------------------------------------------------
availability = {d: 1 for d in range(1, T + 1)}
model1 = build_model(prod, price, coeff, availability, capacity, fixed_cost)

solver = SolverFactory('glpk')
results1 = solver.solve(model1, tee=False)

print("=== RESULTS WITHOUT NO-MAINTENANCE WINDOW ===")
print_solution(model1, results1)
print("-------------------------------------------------\n")

# ---------------------------------------------------------------------
# Solve model with "no-maintenance window" [150..200]
# ---------------------------------------------------------------------
availability = {d: 0 if 150 <= d <= 200 else 1 for d in range(1, T + 1)}
model2 = build_model(prod, price, coeff, availability, capacity, fixed_cost)

results2 = solver.solve(model2, tee=False)

print("=== RESULTS WITH NO-MAINTENANCE WINDOW [150..200] ===")
print_solution(model2, results2)
print("-------------------------------------------------")